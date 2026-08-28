// Shared device-selection helper for the self-contained Vulkan compute helpers
// (deform_conv_vk.cpp, decimate_qem_vk.cpp).
//
// Both helpers create their own VkInstance/VkDevice rather than borrowing ggml's, so each used
// to run an independent "best GPU" ranking. On a multi-GPU box that put the model on one device
// (per --gpu) and the helpers on another, and the two helpers could even disagree with each
// other -- wrong device plus VRAM pressure on a card the user did not pick.
//
// Policy: follow the device the ggml model backend chose. trellis_model.cpp records its PCI bus
// id in g_gpu_pci_id; we match candidates against it via VK_EXT_pci_bus_info. If the id is
// unknown (ggml reported none, or the driver lacks the extension) we fall back to the previous
// rank-then-heap heuristic so behavior is unchanged on single-GPU systems.
#pragma once

#include <vulkan/vulkan.h>

#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "trellis_args.h"   // g_gpu_pci_id

namespace trellis {
namespace vkpick {

// "domain:bus:device.function", lower-case hex -- the format ggml documents for device_id.
inline std::string pci_id_of(VkPhysicalDevice pd) {
    // Probe for the extension first: querying an unsupported struct leaves it zeroed, which
    // would alias to the valid address 0000:00:00.0.
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr) != VK_SUCCESS || n == 0)
        return {};
    std::vector<VkExtensionProperties> ext(n);
    if (vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, ext.data()) != VK_SUCCESS) return {};
    bool have = false;
    for (const auto& e : ext)
        if (std::strcmp(e.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0) { have = true; break; }
    if (!have) return {};

    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &pci;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x",
                  pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
    return std::string(buf);
}

// Rank used only when the model's PCI id is unavailable: real GPUs first, discrete over
// integrated. Never a software rasterizer (llvmpipe reports system RAM as a huge device-local
// heap and would win a size-only heuristic while being CPU-slow).
inline int type_rank(VkPhysicalDeviceType t) {
    return t == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 3
         : t == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
         : t == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU    ? 1 : 0;
}

inline VkDeviceSize device_local_heap(VkPhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkDeviceSize heap = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            heap = heap > mp.memoryHeaps[i].size ? heap : mp.memoryHeaps[i].size;
    return heap;
}

// True if `pd` is the device the model backend is on. False when the id is unknown either side,
// so callers must treat a all-false result as "fall back to the heuristic".
// Compared case-insensitively: ggml's Vulkan backend formats with lower-case %02x (same as
// pci_id_of above), but its CUDA backend stores cudaDeviceGetPCIBusId() verbatim, which yields
// UPPER-case hex letters (e.g. "0000:C1:00.0"). A CUDA build does not compile these Vulkan
// helpers today, so that path is unreachable -- but matching case-insensitively costs nothing
// and keeps this correct if the backends are ever built together.
inline bool matches_model_device(VkPhysicalDevice pd) {
    if (g_gpu_pci_id.empty()) return false;
    const std::string id = pci_id_of(pd);
    if (id.empty() || id.size() != g_gpu_pci_id.size()) return false;
    for (size_t i = 0; i < id.size(); ++i) {
        const unsigned char a = (unsigned char) id[i], b = (unsigned char) g_gpu_pci_id[i];
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

// Log which policy was used, once per helper, so a mismatch is diagnosable in the field.
inline void log_choice(const char* tag, const char* name, bool by_pci) {
    fprintf(stderr, "[%s] using %s (%s)\n", tag, name,
            by_pci ? "matched model device by PCI id" : "heuristic: no PCI id from ggml");
}

}  // namespace vkpick
}  // namespace trellis
