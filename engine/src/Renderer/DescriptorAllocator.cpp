#include "Momo/Renderer/DescriptorAllocator.h"
#include "Momo/Logging/Logger.h"

namespace Momo::Renderer {
void DescriptorAllocator::Init(vk::Device device, const DescriptorPoolSizes& poolSizes) {
    m_Device = device;
    m_PoolSizes = poolSizes;
    CreateDescriptorPool();
}

// Allocates a descriptor set from the pool. 
vk::DescriptorSet DescriptorAllocator::AllocateDescriptorSet(vk::DescriptorSetLayout layout) {
    if (!m_DescriptorPool) {
        CreateDescriptorPool();
    }

    vk::DescriptorSetAllocateInfo allocInfo(
        m_DescriptorPool,
        1,
        &layout
    );

    try {
        auto descriptorSets = m_Device.allocateDescriptorSets(allocInfo);
        if (descriptorSets.empty()) {
            throw std::runtime_error("Failed to allocate descriptor set!");
        }
        return descriptorSets[0];
    } catch (const vk::OutOfPoolMemoryError& e) {
        LOG_ERROR("DescriptorAllocator", "Out of pool memory: {}", e.what());
        throw e; // TODO: Allocate a new pool and retry allocation
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to allocate descriptor set: ") + e.what());
    }
}

void DescriptorAllocator::Destroy() {
    if (m_DescriptorPool) {
        m_Device.destroyDescriptorPool(m_DescriptorPool);
        m_DescriptorPool = nullptr;
    }
}

void DescriptorAllocator::CreateDescriptorPool() {
    vk::DescriptorPoolCreateInfo poolInfo({}, m_PoolSizes.maxSets, m_PoolSizes.sizes);
    m_DescriptorPool = m_Device.createDescriptorPool(poolInfo);
}
} // namespace Momo::Renderer
