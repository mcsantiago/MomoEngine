#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <optional>

namespace Momo::Renderer {
struct DescriptorAllocator {
public:
    void Init(vk::Device device, std::optional<uint32_t> poolSize);
    vk::DescriptorSet AllocateDescriptorSet(vk::DescriptorSetLayout layout);
    void Destroy();

private:
    vk::Device m_Device;
    vk::DescriptorPool m_DescriptorPool;
    uint32_t m_PoolSize = 32;

    void CreateDescriptorPool();
};
} // namespace Momo::Renderer
