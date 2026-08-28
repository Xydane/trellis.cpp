// Validate the custom deformable conv kernel against torchvision (tools/ref_deform.py),
// plus a self-contained GPU-vs-CPU bit-equality check that needs no reference data.
//   trellis-test-deform <ref_dir> [gpu]   compare against the dumped torchvision golden
//   trellis-test-deform --selftest         GPU vs CPU on procedural input (no files needed)
#include "deform_conv.h"
#include "npy.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <string>
#include <vector>

// The GPU kernels (CUDA/HIP --fmad=false, Vulkan `precise`) and the CPU port
// (-ffp-contract=off) are all pinned against FMA contraction so the BiRefNet matte does not
// depend on the backend. That matters because the matte feeds a bbox reduction over
// `alpha > 0.8f` in preprocess.cpp, where one flipped pixel moves the crop and rescales the
// whole conditioning image. Measured on this case with contraction left ON: 221329 of 262144
// outputs differed (max 3.3e-6); with it off, 0 differed. This guards that pin.
//
// Only meaningful where BOTH kernels exist. A CUDA build does not compile deform_conv_cpu.cpp
// (deform_conv.cu provides deform_conv2d_run and there is no second definition), so the
// comparison is available on the Vulkan build -- which is also the one whose shader numerics
// are hardest to verify by inspection. On CUDA, the equivalent check lives in the standalone
// harness described in the commit that added the pin.
#ifdef TRELLIS_HAVE_DEFORM_CPU
static int selftest() {
    const int Cin = 64, Cout = 64, H = 64, W = 64, K = 3;
    std::mt19937 rng(42);
    std::normal_distribution<float> N(0.f, 1.f);
    std::vector<float> x((size_t)Cin*H*W), off((size_t)2*K*K*H*W), msk((size_t)K*K*H*W),
                       w((size_t)Cout*Cin*K*K), b(Cout);
    for (auto& v : x)   v = N(rng);
    for (auto& v : off) v = N(rng) * 2.f;          // offsets straddle the bilinear boundary guards
    for (auto& v : msk) v = std::fabs(N(rng));
    for (auto& v : w)   v = N(rng) * 0.1f;
    for (auto& v : b)   v = N(rng) * 0.01f;

    std::vector<float> gpu((size_t)Cout*H*W), cpu((size_t)Cout*H*W);
    trellis::deform_conv2d_run(x.data(), Cin, H, W, off.data(), msk.data(), w.data(), b.data(),
                               Cout, K, gpu.data(), 0);
    trellis::deform_conv2d_cpu(x.data(), Cin, H, W, off.data(), msk.data(), w.data(), b.data(),
                               Cout, K, cpu.data(), -1);

    size_t ndiff = 0; double mx = 0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        const double d = std::fabs((double)gpu[i] - (double)cpu[i]);
        if (d != 0) ++ndiff;
        if (d > mx) mx = d;
    }
    // Count matte decisions that would flip -- the quantity that actually reaches the pipeline.
    size_t flips = 0;
    for (size_t i = 0; i < gpu.size(); ++i) {
        const double a = 1.0 / (1.0 + std::exp(-(double)gpu[i]));
        const double c = 1.0 / (1.0 + std::exp(-(double)cpu[i]));
        if ((a > 0.8) != (c > 0.8)) ++flips;
    }
    std::printf("deform selftest Cin=%d Cout=%d %dx%d K=%d: differing=%zu/%zu max|d|=%.3g "
                "matte-flips=%zu\n", Cin, Cout, H, W, K, ndiff, gpu.size(), mx, flips);
    // Note: on a pure-CPU build deform_conv2d_run IS the CPU kernel, so this is trivially 0.
    if (ndiff != 0) {
        std::printf("FAIL: GPU deform is not bit-identical to the CPU port "
                    "(check --fmad=false / -ffp-contract=off / `precise` in deform_conv.comp)\n");
        return 1;
    }
    std::printf("PASS: GPU deform is bit-identical to the CPU port\n");
    return 0;
}
#else
static int selftest() {
    std::printf("deform selftest: unavailable in this build (no CPU reference kernel linked; "
                "deform_conv_cpu.cpp is only compiled on the Vulkan/CPU paths)\n");
    return 0;
}
#endif

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 2) { fprintf(stderr, "usage: %s <ref_dir> [gpu] | --selftest\n", argv[0]); return 1; }
    int gpu = argc > 2 ? atoi(argv[2]) : 1;
    std::string d = argv[1];
    npy::Array x = npy::load(d + "/x.npy"), off = npy::load(d + "/offset.npy"),
               mask = npy::load(d + "/mask.npy"), w = npy::load(d + "/weight.npy"),
               b = npy::load(d + "/bias.npy"), g = npy::load(d + "/out.npy");
    int Cin = (int)x.shape[0], H = (int)x.shape[1], W = (int)x.shape[2];
    int Cout = (int)w.shape[0], K = (int)w.shape[2];
    std::vector<float> out((size_t)Cout * H * W);
    trellis::deform_conv2d_run(x.data.data(), Cin, H, W, off.data.data(), mask.data.data(),
                               w.data.data(), b.data.data(), Cout, K, out.data(), gpu);
    double maxd = 0, gmax = 0, sad = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        maxd = std::max(maxd, (double)std::fabs(out[i] - g.data[i]));
        gmax = std::max(gmax, (double)std::fabs(g.data[i])); sad += std::fabs(out[i] - g.data[i]);
    }
    printf("deform K=%d Cin=%d Cout=%d %dx%d : max|d|=%.4e mean|d|=%.4e rel=%.4e  %s\n",
           K, Cin, Cout, H, W, maxd, sad / out.size(), maxd / gmax, maxd / gmax < 1e-4 ? "OK" : "**");
    return 0;
}
