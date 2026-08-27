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

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DGGML_VULKAN=ON \
        -DGGML_NATIVE=OFF \
    && cmake --build build --target trellis-server trellis-cli ${BUILD_JOBS:+-j ${BUILD_JOBS}}

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
