# trellis.cpp — Vulkan backend container image
#
#   Build:  podman build -t trellis-vulkan .
#   Run:    podman run --rm --device /dev/dri \
#             -v ./models:/models:ro -v ./out:/out:z \
#             trellis-vulkan trellis-cli /assets/goblin.png /out/goblin.glb -m /models
#
#   Server: podman run --rm --device /dev/dri -p 8080:8080 \
#             -v ./models:/models:ro trellis-vulkan
#
#   Check the GPU is visible inside the container:
#             podman run --rm --device /dev/dri trellis-vulkan vulkaninfo --summary
#   Without /dev/dri you get Mesa's llvmpipe (software Vulkan) — correct but slow.
#   NVIDIA hosts: use the CDI hook instead, e.g. --device nvidia.com/gpu=all.
#
# The vendored ggml submodule must be checked out before building:
#   git submodule update --init --recursive
#
# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- builder ----
FROM ubuntu:26.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release
# Leave empty to auto-pick a memory-aware job count (see below).
ARG BUILD_JOBS=

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        pkg-config \
        # Vulkan loader + headers, and the shader toolchain used to compile
        # src/deform_conv.comp and src/decimate_qem.comp into SPIR-V.
        libvulkan-dev \
        glslc \
        glslang-tools \
        spirv-headers \
        spirv-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Fail early with a useful message instead of a confusing CMake error.
RUN test -f thirdparty/ggml/CMakeLists.txt \
    || { echo "ERROR: thirdparty/ggml is empty. Run: git submodule update --init --recursive"; exit 1; }

# ggml's vulkan-shaders-gen forks up to 16 concurrent glslc children *per ninja
# job*, and ninja runs many shader-gen steps at once, so the real fan-out is
# jobs x 16. When that exhausts RAM, fork() fails with ENOMEM and the generator
# only *warns* ("Cannot allocate memory" / "Failed to fork process") and skips
# the shader -- but it still emits the extern declaration, so the build dies much
# later with thousands of confusing "undefined reference to matmul_..._cm1_len"
# link errors. Cap the job count against available memory to keep fork() alive.
#
# /proc/meminfo reports the *host* totals inside a container, so honour the
# cgroup limit (v2 memory.max, then v1 limit_in_bytes) when one is set.
RUN set -eux; \
    jobs="${BUILD_JOBS:-}"; \
    if [ -z "$jobs" ]; then \
        mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); \
        for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do \
            if [ -r "$f" ]; then \
                lim=$(cat "$f"); \
                case "$lim" in \
                    ''|max|*[!0-9]*) ;; \
                    *) lim_kb=$(( lim / 1024 )); \
                       [ "$lim_kb" -lt "$mem_kb" ] && mem_kb="$lim_kb" ;; \
                esac; \
            fi; \
        done; \
        # ~1 job per 2 GiB: covers both a forking glslc fan-out and a big cc1plus.
        jobs=$(( mem_kb / 2097152 )); \
        [ "$jobs" -lt 1 ] && jobs=1; \
        [ "$jobs" -gt "$(nproc)" ] && jobs=$(nproc); \
    fi; \
    echo "building with -j${jobs} (mem_kb=${mem_kb:-n/a}, nproc=$(nproc))"; \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DGGML_VULKAN=ON \
        -DGGML_NATIVE=OFF; \
    # Phase 1: the shader generation, throttled hard. Each ninja step here spawns
    # a vulkan-shaders-gen that itself forks up to 16 glslc children, so keep the
    # outer width small; this is the step that ran out of memory.
    shader_jobs=2; \
    [ "$jobs" -lt 2 ] && shader_jobs="$jobs"; \
    cmake --build build --target ggml-vulkan -j "${shader_jobs}"; \
    # Phase 2: ordinary C++ compilation, full width.
    cmake --build build --target trellis-server trellis-cli -j "${jobs}"

# Guard against the silent-skip failure mode above: every shader the generated
# header declares must also have a definition in the generated source.
RUN set -eu; \
    gen=build/thirdparty/ggml/src/ggml-vulkan; \
    decl=$(grep -c '^extern const uint64_t' "$gen/ggml-vulkan-shaders.hpp"); \
    echo "generated shader symbols declared: $decl"; \
    test "$decl" -gt 2000 || { echo "ERROR: too few Vulkan shaders generated ($decl) -- shader generation was truncated"; exit 1; }

# Stage the binaries + ggml shared libs and point their rpath at $ORIGIN so the
# runtime image needs no LD_LIBRARY_PATH (mirrors the release packaging step).
RUN apt-get update && apt-get install -y --no-install-recommends patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /dist \
    && cp build/trellis-server build/trellis-cli /dist/ \
    && cp -P build/libggml*.so* /dist/ 2>/dev/null || true \
    && find /dist -type f -exec patchelf --set-rpath '$ORIGIN' {} \; 2>/dev/null || true

# ---------------------------------------------------------------- runtime ----
FROM ubuntu:26.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        # Vulkan loader + open-source ICDs (AMD RADV / Intel ANV / NVIDIA users
        # should instead bind-mount the host driver or use the nvidia CDI hook).
        libvulkan1 \
        mesa-vulkan-drivers \
        vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /dist/ /opt/trellis/
COPY --from=builder /src/assets /assets

ENV PATH="/opt/trellis:${PATH}"

# Nothing writes to the image itself; models are mounted read-only.
VOLUME ["/models", "/out"]
WORKDIR /out
EXPOSE 8080

ENTRYPOINT []
CMD ["trellis-server", "--host", "0.0.0.0", "--port", "8080", "--models", "/models"]
