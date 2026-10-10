#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vector>

namespace Momo::Renderer {
// How many sets a pool holds, and how many descriptors of each type in total
struct DescriptorPoolSizes {
    uint32_t maxSets = 0;
    std::vector<vk::DescriptorPoolSize> sizes;
};

struct DescriptorAllocator {
public:
    void Init(vk::Device device, const DescriptorPoolSizes& poolSizes);
    vk::DescriptorSet AllocateDescriptorSet(vk::DescriptorSetLayout layout);
    void Destroy();

private:
    vk::Device m_Device;
    vk::DescriptorPool m_DescriptorPool;
    DescriptorPoolSizes m_PoolSizes;

    void CreateDescriptorPool();
};
} // namespace Momo::Renderer
