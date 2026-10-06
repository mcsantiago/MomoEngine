#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Renderer/VulkanMeshData.h"
#include "Momo/WindowVulkan.h"
#include "Momo/Logging/Logger.h"
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cstring>
#include <set>
#include <algorithm>
#include <array>
#include <vulkan/vulkan_raii.hpp>

#ifdef NDEBUG
    constexpr bool enableValidationLayers = false;
#else
    constexpr bool enableValidationLayers = true;
#endif

namespace 
{
    struct QueueFamilyIndices 
    {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() const 
        {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }

        std::set<uint32_t> UniqueFamilies() const 
        {
            return {graphicsFamily.value(), presentFamily.value()};
        }
    };

    QueueFamilyIndices FindQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::SurfaceKHR& surface)
    {
        QueueFamilyIndices indices;

        std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) 
        {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) 
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, static_cast<VkSurfaceKHR>(*surface));

            if (presentSupport) 
            {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) 
            {
                break;
            }

            i++;
        }

        return indices;
    }

    vk::SurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats) 
        {
            if (availableFormat.format == vk::Format::eR8G8B8A8Srgb && 
                availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) 
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR ChooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
    {
        for (const auto& availablePresentMode : availablePresentModes) 
        {
            // Prefer Mailbox if available and energy is not a concern
            // Otherwise fall back to FIFO which is guaranteed to be available
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) 
            {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D ChooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, uint32_t width, uint32_t height)
    {
        if (capabilities.currentExtent.width != UINT32_MAX) 
        {
            return capabilities.currentExtent;
        } 
        else 
        {
            vk::Extent2D actualExtent = {
                std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
                std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
            };
            return actualExtent;
        }
    }

    static std::vector<char> ReadFile(const std::string& filename)
    {
        LOG_DEBUG("VulkanRenderer", "Reading shader file: {}", filename);
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) 
        {
            LOG_ERROR("VulkanRenderer", "Failed to open file: {}", filename);
            throw std::runtime_error("Failed to open file: " + filename);
        }

        LOG_DEBUG("VulkanRenderer", "File opened successfully");
        std::vector<char> buffer(file.tellg());
        LOG_DEBUG("VulkanRenderer", "File size: {} bytes", buffer.size());
        file.seekg(0);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        LOG_DEBUG("VulkanRenderer", "File read into buffer");
        return buffer;
    }
}

namespace Momo {
namespace Renderer {
    static vk::VertexInputBindingDescription GetBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription(
            0,                          // binding
            sizeof(Geometry::Vertex),             // stride
            vk::VertexInputRate::eVertex // inputRate
        );
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 3> GetAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions = {};
        attributeDescriptions[0] = vk::VertexInputAttributeDescription(
            0,                          // location
            0,                          // binding
            vk::Format::eR32G32B32Sfloat,  // format
            offsetof(Geometry::Vertex, position)  // offset
        );
        attributeDescriptions[1] = vk::VertexInputAttributeDescription(
            1,                          // location
            0,                          // binding
            vk::Format::eR32G32B32Sfloat, // format
            offsetof(Geometry::Vertex, normal)   // offset
        );
        attributeDescriptions[2] = vk::VertexInputAttributeDescription(
            2,                          // location
            0,                          // binding
            vk::Format::eR32G32Sfloat, // format
            offsetof(Geometry::Vertex, texCoord)   // offset
        );
        return attributeDescriptions;
    }

    void VulkanRenderer::Init(const IWindow& window)
    {
        m_Window = &window;
        CreateInstance();
        LOG_DEBUG("VulkanRenderer", "Creating window surface...");
        CreateSurface();
        LOG_DEBUG("VulkanRenderer", "Picking physical device...");
        PickPhysicalDevice();
        LOG_DEBUG("VulkanRenderer", "Creating logical device...");
        CreateLogicalDevice();
        LOG_DEBUG("VulkanRenderer", "Checking timestamp support...");
        CheckTimestampSupport();        // TODO: Maybe this should be checked only if GPU timing is enabled in the config
        LOG_DEBUG("VulkanRenderer", "Createing timestamp query pool...");
        CreateTimestampQueryPool();
        LOG_DEBUG("VulkanRenderer", "Creating swap chain...");
        CreateSwapChain();
        LOG_DEBUG("VulkanRenderer", "Creating image views...");
        CreateImageView();
        LOG_DEBUG("VulkanRenderer", "Creating descriptor set layout...");
        CreateDescriptorSetLayout();
        LOG_DEBUG("VulkanRenderer", "Creating graphics pipeline...");
        CreateGraphicsPipeline();
        LOG_DEBUG("VulkanRenderer", "Creating command pool...");
        CreateCommandPool();
        LOG_DEBUG("VulkanRenderer", "Creating command buffer...");
        CreateCommandBuffer();
        LOG_DEBUG("VulkanRenderer", "Creating synchronization objects...");
        CreateSyncObjects();
        LOG_DEBUG("VulkanRenderer", "Creating texture sampler...");
        CreateTextureSampler();
        m_DescriptorAllocator.Init(*m_Device, std::nullopt);
    }

    void VulkanRenderer::CheckTimestampSupport()
    {
        m_TimestampPeriodNs = m_PhysicalDevice->getProperties().limits.timestampPeriod;
        auto families = m_PhysicalDevice->getQueueFamilyProperties();
        uint32_t validBits = families[m_GraphicsQueueFamilyIdx].timestampValidBits;
        if (validBits == 0) {
            LOG_WARN("VulkanRenderer", "GPU does not support timestamps (validBits == 0). GPU timing disabled.");
            m_GpuTimingEnabled = false;
            return;
        }
        m_GpuTimingEnabled = true;
        LOG_INFO("VulkanRenderer", "GPU timing enabled. Timestamp period: {} ns, valid bits: {}", m_TimestampPeriodNs, validBits);
    }

    void VulkanRenderer::CreateTimestampQueryPool()
    {
        if (!m_GpuTimingEnabled) {
            LOG_DEBUG("VulkanRenderer", "Skipping timestamp query pool creation because GPU timing is disabled.");
            return;
        }
        vk::QueryPoolCreateInfo queryPoolInfo{};
        queryPoolInfo.queryType = vk::QueryType::eTimestamp;
        queryPoolInfo.queryCount = 2; // Start and end timestamps
        m_TimestampQueryPool = vk::raii::QueryPool(*m_Device, queryPoolInfo);
        LOG_DEBUG("VulkanRenderer", "Created timestamp query pool with {} queries.", queryPoolInfo.queryCount);
    }

    void VulkanRenderer::CreateTextureSampler()
    {
        LOG_DEBUG("VulkanRenderer", "Creating texture sampler, descriptor set layout, and descriptor pool...");
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = VK_FALSE;
        // samplerInfo.maxAnisotropy = 16;
        samplerInfo.maxLod = 0;
        // samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        // samplerInfo.unnormalizedCoordinates = VK_FALSE;
        // samplerInfo.compareEnable = VK_FALSE;
        // samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;

        m_TextureSampler = vk::raii::Sampler(*m_Device, samplerInfo);

    }

    void VulkanRenderer::CreateDescriptorSetLayout()
    {
        vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 0;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        vk::DescriptorSetLayoutBinding materialLayoutBinding{};
        materialLayoutBinding.binding = 1;
        materialLayoutBinding.descriptorCount = 1;
        materialLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        materialLayoutBinding.pImmutableSamplers = nullptr;
        materialLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
            samplerLayoutBinding,
            materialLayoutBinding
        };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        m_DescriptorSetLayout = vk::raii::DescriptorSetLayout(*m_Device, layoutInfo);
    }

    void VulkanRenderer::Shutdown()
    {
        // RAII wrappers (vk::raii::Instance) handle destruction automatically
        // Explicitly reset the optional to destroy the instance now
        // This also implicitly destroys the VkPhysicalDevice so no need to set it here
        m_Device->waitIdle();
        m_DescriptorAllocator.Destroy();
    }

    vk::raii::CommandBuffer VulkanRenderer::BeginSingleUseCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo(
            *m_CommandPool,                      // commandPool
            vk::CommandBufferLevel::ePrimary,   // level - can be submitted directly to queue
            1                                    // commandBufferCount
        );

        auto commandBuffers = vk::raii::CommandBuffers(m_Device.value(), allocInfo);
        auto commandBuffer = std::move(commandBuffers[0]);
        commandBuffer.begin({
            vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        });
        LOG_INFO("VulkanRenderer", "Command buffer allocated");
        return commandBuffer;
    }

    void VulkanRenderer::EndSingleUseCommandBuffer(vk::raii::CommandBuffer& commandBuffer)
    {
        commandBuffer.end();
        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandBuffer;
        m_GraphicsQueue->submit(submitInfo);
        m_GraphicsQueue->waitIdle();
    }

    void VulkanRenderer::BeginFrame(uint32_t imageIndex)
    {
        // Implementation for beginning a frame
        LOG_DEBUG("VulkanRenderer", "Beginning a frame...");
        
        // Reset the fence to indicate that the GPU is now using it for the current frame
        m_Device->resetFences(**m_InFlightFence);

        // Record command buffer for the acquired image
        m_CommandBuffer->reset();
        m_CommandBuffer->begin({});

        m_CommandBuffer->resetQueryPool(**m_TimestampQueryPool, 0, 2);
        if (m_GpuTimingEnabled) {
            m_CommandBuffer->writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, **m_TimestampQueryPool, 0);
        }

        vk::ImageMemoryBarrier2 toColorAttachment{};
        toColorAttachment.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        toColorAttachment.srcAccessMask = {};
        toColorAttachment.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        toColorAttachment.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        toColorAttachment.oldLayout = vk::ImageLayout::eUndefined;
        toColorAttachment.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        toColorAttachment.image = m_Swapchain->getImages()[imageIndex];
        toColorAttachment.subresourceRange = {
            vk::ImageAspectFlagBits::eColor, // aspectMask
            0,                                // baseMipLevel
            1,                                // levelCount
            0,                                // baseArrayLayer
            1                                 // layerCount
        };

        vk::ImageMemoryBarrier2 toDepthAttachment{};
        toDepthAttachment.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests;
        toDepthAttachment.srcAccessMask = {};
        toDepthAttachment.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        toDepthAttachment.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        toDepthAttachment.oldLayout = vk::ImageLayout::eUndefined;
        toDepthAttachment.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        toDepthAttachment.image = *m_DepthBuffer->image;
        toDepthAttachment.subresourceRange = {
            vk::ImageAspectFlagBits::eDepth, // aspectMask
            0,                                // baseMipLevel
            1,                                // levelCount
            0,                                // baseArrayLayer
            1                                 // layerCount
        };

        vk::DependencyInfo dependencyInfo{};
        dependencyInfo.imageMemoryBarrierCount = 2;
        vk::ImageMemoryBarrier2 imageBarriers[] = { toColorAttachment, toDepthAttachment };
        dependencyInfo.pImageMemoryBarriers = imageBarriers;

        m_CommandBuffer->pipelineBarrier2(dependencyInfo);

        // Begin rendering
        vk::ClearValue clearColor = vk::ClearValue().setColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        vk::RenderingAttachmentInfo colorAttachment{};
        colorAttachment.imageView = *m_SwapchainImageViews[imageIndex];
        colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.clearValue = clearColor;

        vk::RenderingAttachmentInfo depthAttachment{};
        depthAttachment.imageView = *m_DepthBuffer->imageView;
        depthAttachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachment.clearValue = vk::ClearValue().setDepthStencil({1.0f, 0});

        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D({0,0}, m_SwapchainExtent);
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;
        renderingInfo.pDepthAttachment = &depthAttachment;

        m_CommandBuffer->beginRendering(renderingInfo);
    }

    void VulkanRenderer::EndFrame(uint32_t imageIndex)
    {
        // Implementation for ending a frame
        LOG_DEBUG("VulkanRenderer", "Ending a frame...");

        m_CommandBuffer->endRendering();

        // Transition the swapchain image to present layout
        vk::ImageMemoryBarrier2 toPresent{};
        toPresent.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        toPresent.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        toPresent.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        toPresent.dstAccessMask = {};
        toPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        toPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
        toPresent.image = m_Swapchain->getImages()[imageIndex];
        toPresent.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        toPresent.subresourceRange.baseMipLevel = 0;
        toPresent.subresourceRange.levelCount = 1;
        toPresent.subresourceRange.baseArrayLayer = 0;
        toPresent.subresourceRange.layerCount = 1;

        vk::DependencyInfo presentDependency{};
        presentDependency.imageMemoryBarrierCount = 1;
        presentDependency.pImageMemoryBarriers = &toPresent;

        m_CommandBuffer->pipelineBarrier2(presentDependency);
        if (m_GpuTimingEnabled) {
            m_CommandBuffer->writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, **m_TimestampQueryPool, 1);
        }
        m_CommandBuffer->end();

        // Submit and present
        vk::Semaphore waitSemaphores[]   = { **m_ImageAvailableSemaphore };
        vk::Semaphore signalSemaphores[] = { *m_RenderFinishedSemaphores[imageIndex] };
        vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, **m_CommandBuffer, signalSemaphores);
        m_GraphicsQueue->submit(submitInfo, **m_InFlightFence);
        if (m_GpuTimingEnabled) {
            m_HasTimestampQueryPoolResults = true;
        }

        vk::PresentInfoKHR presentInfo(signalSemaphores, **m_Swapchain, imageIndex);
        auto presentResult = m_PresentQueue->presentKHR(presentInfo);
        
        if (presentResult == vk::Result::eSuboptimalKHR) {
            m_SubOptimal = true;
        }
    }

    void VulkanRenderer::RenderFrame(VulkanModelData& modelData, glm::mat4 viewMatrix, glm::mat4 modelMatrix)
    {
        // Implementation for rendering a single frame using Vulkan
        LOG_DEBUG("VulkanRenderer", "Rendering a frame...");
        if (m_SubOptimal) {
            LOG_DEBUG("VulkanRenderer", "Swapchain is suboptimal. Consider recreating swapchain.");
            RecreateSwapchain();
        }

        // Wait for fences
        while (vk::Result::eTimeout == 
            m_Device->waitForFences(**m_InFlightFence, VK_TRUE, UINT64_MAX));

        
        // Acquire the next image from the swapchain
        try {
            auto [acquireResult, imageIndex] = m_Swapchain->acquireNextImage(
                UINT64_MAX, **m_ImageAvailableSemaphore, nullptr);
            if (acquireResult == vk::Result::eSuboptimalKHR) {
                m_SubOptimal = true;
            }
            // Get last frame's timestamps if GPU timing is enabled
            if (m_GpuTimingEnabled && m_HasTimestampQueryPoolResults) {
                auto [result, ticks] = m_TimestampQueryPool->getResults<uint64_t>(
                0,                          // firstQuery: start at slot 0
                2,                          // queryCount: read slots 0 and 1
                2 * sizeof(uint64_t),       // dataSize: total bytes for the whole output (16)
                sizeof(uint64_t),           // stride: bytes from one result to the next (8)
                vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

                if (result == vk::Result::eSuccess) {
                    uint64_t elapsedTicks = ticks[1] - ticks[0];
                    m_LastGpuFrameMs = elapsedTicks * m_TimestampPeriodNs / 1'000'000.0f;
                    LOG_INFO("VulkanRenderer", "GPU frame time: {} ms", m_LastGpuFrameMs);
                }
            }
            BeginFrame(imageIndex);

            // Draw
            m_CommandBuffer->bindPipeline(vk::PipelineBindPoint::eGraphics, **m_GraphicsPipeline);
            m_CommandBuffer->setViewport(0, vk::Viewport(
                0.0f, 0.0f, 
                static_cast<float>(m_SwapchainExtent.width), 
                static_cast<float>(m_SwapchainExtent.height), 
                0.0f, 1.0f));

            m_CommandBuffer->setScissor(0, vk::Rect2D({0, 0}, m_SwapchainExtent));

            for (const auto& mesh : modelData.meshes) {
                m_CommandBuffer->setCullMode(mesh.materialData.doubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
                m_CommandBuffer->bindDescriptorSets(
                        vk::PipelineBindPoint::eGraphics, 
                        **m_PipelineLayout, 
                        0, 
                        {mesh.materialData.descriptorSet}, 
                        nullptr
                );
                m_CommandBuffer->bindVertexBuffers(0, *mesh.gpuMesh.vertexBuffer.buffer, {0});
                m_CommandBuffer->bindIndexBuffer(*mesh.gpuMesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);
                PushConstantData pushConstantData;
                pushConstantData.projectionMatrix = m_ProjectionMatrix;
                pushConstantData.viewMatrix = viewMatrix;
                pushConstantData.modelMatrix = modelMatrix * mesh.localTransform; // Move to GPU..?
                m_CommandBuffer->pushConstants<PushConstantData>(
                    **m_PipelineLayout, 
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 
                    0,
                    pushConstantData
                );
                m_CommandBuffer->drawIndexed(mesh.gpuMesh.indexBuffer.indexCount, 1, 0, 0, 0); // Draw a quad using indices
            }

            EndFrame(imageIndex);
        } catch (const vk::OutOfDateKHRError& e) {
            LOG_DEBUG("VulkanRenderer", "Swapchain is out of date. Need to recreate swapchain.");
            RecreateSwapchain();
        }

    }

    uint32_t VulkanRenderer::FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memProperties = m_PhysicalDevice->getMemoryProperties();
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type!");
    }

    AllocatedBuffer VulkanRenderer::CreateVertexBuffer(const std::vector<Geometry::Vertex>& vertices)
    {
        return CreateBuffer(vertices.data(), sizeof(Geometry::Vertex) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer, static_cast<uint32_t>(vertices.size()));
    }

    AllocatedBuffer VulkanRenderer::CreateIndexBuffer(const std::vector<uint32_t>& indices)
    {
        return CreateBuffer(indices.data(), sizeof(uint32_t) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer, static_cast<uint32_t>(indices.size()));
    }

    AllocatedImage VulkanRenderer::CreateAndSubmitTextureImage(const Assets::TextureData& texture)
    {
        auto stagingBuffer = CreateTextureStagingBuffer(texture);
        auto textureImage = CreateTextureImage(texture);
        SubmitTextureImage(stagingBuffer, textureImage);
        return textureImage;
    }

    VulkanMaterialData VulkanRenderer::CreateVulkanMaterialData(const AllocatedImage& baseColorTexture, const MaterialParams& materialParams, bool doubleSided)
    {
        auto ubo = CreateBuffer(&materialParams, sizeof(MaterialParams), vk::BufferUsageFlagBits::eUniformBuffer, 0);
        auto descriptorSet = m_DescriptorAllocator.AllocateDescriptorSet(**m_DescriptorSetLayout);
        WriteDescriptorSet(descriptorSet, ubo, baseColorTexture);
        return VulkanMaterialData{ 
            .doubleSided=doubleSided,
            .descriptorSet=descriptorSet, 
            .paramsBuffer=std::move(ubo)
        };
    }

    void VulkanRenderer::WriteDescriptorSet(const vk::DescriptorSet descriptorSet, const AllocatedBuffer& ubo,  const AllocatedImage& textureImage)
    {
        vk::DescriptorImageInfo imageInfo = vk::DescriptorImageInfo(
            *m_TextureSampler,
            textureImage.imageView,
            vk::ImageLayout::eShaderReadOnlyOptimal
        );

        vk::WriteDescriptorSet descriptorWrite = vk::WriteDescriptorSet();
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.pImageInfo = &imageInfo;

        vk::DescriptorBufferInfo bufferInfo = vk::DescriptorBufferInfo {
            *ubo.buffer, 0, sizeof(MaterialParams)
        };

        vk::WriteDescriptorSet uboDescriptorWrite = vk::WriteDescriptorSet();
        uboDescriptorWrite.dstSet = descriptorSet;
        uboDescriptorWrite.dstBinding = 1;
        uboDescriptorWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboDescriptorWrite.descriptorCount = 1;
        uboDescriptorWrite.pBufferInfo = &bufferInfo;

        m_Device->updateDescriptorSets({descriptorWrite, uboDescriptorWrite}, {});
    }

    AllocatedImage VulkanRenderer::CreateTextureImage(const Assets::TextureData& texture)
    {
        auto image = vk::raii::Image(*m_Device, vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            vk::Format::eR8G8B8A8Srgb,
            vk::Extent3D{ texture.width, texture.height, 1 },
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            vk::ImageLayout::eUndefined
        ));
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, 
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto imageMemory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(
            memRequirements.size, memoryTypeIndex
        ));
        image.bindMemory(*imageMemory, 0);

        vk::ImageViewCreateInfo viewInfo(
            {},
            *image,
            vk::ImageViewType::e2D,
            vk::Format::eR8G8B8A8Srgb,
            {},
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor,
                0, 1,
                0, 1
            )
        );
        auto imageView = vk::raii::ImageView(*m_Device, viewInfo);

        return AllocatedImage{ std::move(imageMemory), std::move(image), std::move(imageView), texture.width, texture.height };
    }

    void VulkanRenderer::SubmitTextureImage(const AllocatedBuffer& stagingBuffer, const AllocatedImage& textureImage) 
    {
        auto commandBuffer = BeginSingleUseCommandBuffer();
        auto image = &textureImage.image;

        vk::ImageMemoryBarrier2 beforeCopyBarrier{};
        beforeCopyBarrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        beforeCopyBarrier.srcAccessMask = {};
        beforeCopyBarrier.dstStageMask = vk::PipelineStageFlagBits2::eCopy;
        beforeCopyBarrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        beforeCopyBarrier.oldLayout = vk::ImageLayout::eUndefined;
        beforeCopyBarrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
        beforeCopyBarrier.image = *image;
        beforeCopyBarrier.subresourceRange = {
            vk::ImageAspectFlagBits::eColor, // aspectMask
            0,                                // baseMipLevel
            1,                                // levelCount
            0,                                // baseArrayLayer
            1                                 // layerCount
        };
        auto beforeCopyDependencyInfo = vk::DependencyInfo();
        vk::ImageMemoryBarrier2 imageBarriers[] = { beforeCopyBarrier };
        beforeCopyDependencyInfo.setImageMemoryBarrierCount(1);
        beforeCopyDependencyInfo.setImageMemoryBarriers(imageBarriers);

        vk::ImageMemoryBarrier2 afterCopyBarrier{};
        afterCopyBarrier.srcStageMask = vk::PipelineStageFlagBits2::eCopy;
        afterCopyBarrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        afterCopyBarrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        afterCopyBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        afterCopyBarrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        afterCopyBarrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        afterCopyBarrier.image = *image;
        afterCopyBarrier.subresourceRange = {
            vk::ImageAspectFlagBits::eColor, // aspectMask
            0,                                // baseMipLevel
            1,                                // levelCount
            0,                                // baseArrayLayer
            1                                 // layerCount
        };
        auto afterCopyDependencyInfo = vk::DependencyInfo();
        vk::ImageMemoryBarrier2 imageBarriers2[] = { afterCopyBarrier };
        afterCopyDependencyInfo.setImageMemoryBarrierCount(1);
        afterCopyDependencyInfo.setImageMemoryBarriers(imageBarriers2);

        commandBuffer.pipelineBarrier2(beforeCopyDependencyInfo);
        // copy buffer to image 
        vk::BufferImageCopy copyRegion(
            0,
            0,
            0,
            vk::ImageSubresourceLayers(
                vk::ImageAspectFlagBits::eColor,
                0,
                0,
                1
            ),
            vk::Offset3D{ 0, 0, 0 },
            vk::Extent3D{ textureImage.width, textureImage.height, 1 }
        );
        commandBuffer.copyBufferToImage(
            stagingBuffer.buffer,
            *image,
            vk::ImageLayout::eTransferDstOptimal,
            copyRegion
        );

        commandBuffer.pipelineBarrier2(afterCopyDependencyInfo);
        EndSingleUseCommandBuffer(commandBuffer);
    }

    AllocatedBuffer VulkanRenderer::CreateTextureStagingBuffer(const Assets::TextureData& texture)
    {
        return CreateBuffer(texture.pixelData.data(), sizeof(uint8_t) * texture.pixelData.size(), vk::BufferUsageFlagBits::eTransferSrc, static_cast<uint32_t>(texture.pixelData.size()));
    }

    AllocatedImage VulkanRenderer::CreateDepthBuffer()
    {
        auto image = vk::raii::Image(*m_Device, vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            m_DepthFormat,
            vk::Extent3D{ m_SwapchainExtent.width, m_SwapchainExtent.height, 1 },
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            vk::ImageLayout::eUndefined
        ));
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, 
            vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto imageMemory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(
            memRequirements.size, memoryTypeIndex
        ));
        image.bindMemory(*imageMemory, 0);
        vk::ImageViewCreateInfo viewInfo(
            {},
            *image,
            vk::ImageViewType::e2D,
            m_DepthFormat,
            {},
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eDepth,
                0, 1,
                0, 1
            )
        );
        auto imageView = vk::raii::ImageView(*m_Device, viewInfo);

        return AllocatedImage{ std::move(imageMemory), std::move(image), std::move(imageView) };
    }

    AllocatedBuffer VulkanRenderer::CreateBuffer(const void* data, vk::DeviceSize size, vk::BufferUsageFlags usage, uint32_t indexCount) 
    {
        auto buffer = vk::raii::Buffer(*m_Device, vk::BufferCreateInfo(
            {}, size, usage,
            vk::SharingMode::eExclusive
        ));
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, 
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        auto bufferMemory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(
            memRequirements.size, memoryTypeIndex
        ));
        buffer.bindMemory(*bufferMemory, 0);
        void* mappedData = bufferMemory.mapMemory(0, size);
        memcpy(mappedData, data, (size_t)size);
        bufferMemory.unmapMemory();
        return AllocatedBuffer{ std::move(bufferMemory), std::move(buffer), indexCount };
    }


    void VulkanRenderer::CreateGraphicsPipeline()
    {
        // You'll need a new graphics pipeline for every shader

        // ═══════════════════════════════════════════════════════════
        // SHADER STAGES
        // ═══════════════════════════════════════════════════════════
        LOG_DEBUG("VulkanRenderer", "Loading shader modules...");
        auto vertShaderCode = ReadFile("shaders/triangle.vert.spv");
        auto fragShaderCode = ReadFile("shaders/triangle.frag.spv");
        vk::raii::ShaderModule vertShaderModule = CreateShaderModule(vertShaderCode);
        vk::raii::ShaderModule fragShaderModule = CreateShaderModule(fragShaderCode);

        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "vertMain"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "fragMain")
        };
        LOG_DEBUG("VulkanRenderer", "Shader modules created and stages configured");

        // ═══════════════════════════════════════════════════════════
        // 1. VERTEX INPUT - Describes vertex data format
        // ═══════════════════════════════════════════════════════════
        auto bindingDescription = GetBindingDescription();
        auto attributeDescriptions = GetAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
            {},        // flags
            1, &bindingDescription, // vertexBindingDescriptions
            static_cast<uint32_t>(attributeDescriptions.size()), attributeDescriptions.data()  // vertexAttributeDescriptions
        );
        LOG_DEBUG("VulkanRenderer", "Vertex input state configured (no vertex data)");

        // ═══════════════════════════════════════════════════════════
        // 2. INPUT ASSEMBLY - How to interpret vertex data
        // ═══════════════════════════════════════════════════════════
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
            {},                                    // flags
            vk::PrimitiveTopology::eTriangleList, // Every 3 vertices form a triangle
            VK_FALSE                               // primitiveRestartEnable
        );
        LOG_DEBUG("VulkanRenderer", "Input assembly state configured (triangle list)");

        // ═══════════════════════════════════════════════════════════
        // 3. VIEWPORT & SCISSOR - Render region on screen
        // ═══════════════════════════════════════════════════════════
        // Using dynamic state, so we just specify the count here
        vk::PipelineViewportStateCreateInfo viewportState(
            {},
            1, nullptr,  // viewportCount (actual viewport set dynamically)
            1, nullptr   // scissorCount (actual scissor set dynamically)
        );
        LOG_DEBUG("VulkanRenderer", "Viewport state configured (dynamic viewport and scissor)");

        // ═══════════════════════════════════════════════════════════
        // 4. RASTERIZER - Converts geometry into fragments
        // ═══════════════════════════════════════════════════════════
        vk::PipelineRasterizationStateCreateInfo rasterizer(
            {},                               // flags
            VK_FALSE,                         // depthClampEnable
            VK_FALSE,                         // rasterizerDiscardEnable
            vk::PolygonMode::eFill,          // polygonMode - fill triangles
            {},                                      // cullMode - set dynamically per mesh
            vk::FrontFace::eCounterClockwise,       // frontFace
            VK_FALSE,                         // depthBiasEnable
            0.0f,                             // depthBiasConstantFactor
            0.0f,                             // depthBiasClamp
            0.0f,                             // depthBiasSlopeFactor
            1.0f                              // lineWidth
        );
        LOG_DEBUG("VulkanRenderer", "Rasterizer state configured");

        // ═══════════════════════════════════════════════════════════
        // 5. MULTISAMPLING - Anti-aliasing (disabled for now)
        // ═══════════════════════════════════════════════════════════
        vk::PipelineMultisampleStateCreateInfo multisampling(
            {},                              // flags
            vk::SampleCountFlagBits::e1,    // rasterizationSamples - no multisampling
            VK_FALSE,                        // sampleShadingEnable
            1.0f,                            // minSampleShading
            nullptr,                         // pSampleMask
            VK_FALSE,                        // alphaToCoverageEnable
            VK_FALSE                         // alphaToOneEnable
        );
        LOG_DEBUG("VulkanRenderer", "Multisampling state configured (disabled)");

        // ═══════════════════════════════════════════════════════════
        // 6. DEPTH/STENCIL - Not needed for 2D triangle
        // ═══════════════════════════════════════════════════════════
        // Pass nullptr to pDepthStencilState

        // ═══════════════════════════════════════════════════════════
        // 7. COLOR BLENDING - How fragment colors combine with framebuffer
        // ═══════════════════════════════════════════════════════════
        vk::PipelineColorBlendAttachmentState colorBlendAttachment(
            VK_FALSE,                        // blendEnable - just overwrite
            vk::BlendFactor::eOne,          // srcColorBlendFactor
            vk::BlendFactor::eZero,         // dstColorBlendFactor
            vk::BlendOp::eAdd,              // colorBlendOp
            vk::BlendFactor::eOne,          // srcAlphaBlendFactor
            vk::BlendFactor::eZero,         // dstAlphaBlendFactor
            vk::BlendOp::eAdd,              // alphaBlendOp
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA  // colorWriteMask
        );
        LOG_DEBUG("VulkanRenderer", "Color blending state configured (no blending)");

        vk::PipelineColorBlendStateCreateInfo colorBlending(
            {},                              // flags
            VK_FALSE,                        // logicOpEnable
            vk::LogicOp::eCopy,             // logicOp
            1, &colorBlendAttachment,       // attachmentCount, pAttachments
            {0.0f, 0.0f, 0.0f, 0.0f}        // blendConstants
        );
        LOG_DEBUG("VulkanRenderer", "Color blend state configured");

        // ═══════════════════════════════════════════════════════════
        // 8. DYNAMIC STATE - Properties that can change without pipeline recreation
        // ═══════════════════════════════════════════════════════════
        std::array<vk::DynamicState, 3> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor,
            vk::DynamicState::eCullMode, 
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);
        LOG_DEBUG("VulkanRenderer", "Dynamic state configured (viewport and scissor)");

        // ═══════════════════════════════════════════════════════════
        // 9. PIPELINE LAYOUT - Describes shader resource bindings (uniforms, etc.)
        // ═══════════════════════════════════════════════════════════
        auto pushConstantRange = vk::PushConstantRange(
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, // stageFlags
            0,                                 // offset
            static_cast<uint32_t>(sizeof(PushConstantData))                  // size
        );

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo(
            {},         // flags
            1, &**m_DescriptorSetLayout, // setLayoutCount, pSetLayouts (descriptor sets)
            1, &pushConstantRange // pushConstantRangeCount, pPushConstantRanges
        );
        m_PipelineLayout = vk::raii::PipelineLayout(m_Device.value(), pipelineLayoutInfo);
        LOG_DEBUG("VulkanRenderer", "Pipeline layout created");

        // ═══════════════════════════════════════════════════════════
        // 10. DYNAMIC RENDERING INFO (Vulkan 1.3 - replaces render passes)
        // ═══════════════════════════════════════════════════════════
        vk::Format colorFormat = m_SwapchainImageFormat.format;
        vk::PipelineRenderingCreateInfo renderingInfo(
            {},                           // flags
            1, &colorFormat,             // viewMask, colorAttachmentCount, pColorAttachmentFormats
            m_DepthFormat,      // depthAttachmentFormat
            vk::Format::eUndefined       // stencilAttachmentFormat
        );
        LOG_DEBUG("VulkanRenderer", "Dynamic rendering info configured");

        vk::PipelineDepthStencilStateCreateInfo depthStencilInfo(
            {},                          // flags
            VK_TRUE,                     // depthTestEnable
            VK_TRUE,                     // depthWriteEnable
            vk::CompareOp::eLess,        // depthCompareOp
            VK_FALSE,                    // depthBoundsTestEnable
            VK_FALSE                     // stencilTestEnable
        );

        // ═══════════════════════════════════════════════════════════
        // FINAL ASSEMBLY - Bundle everything into the graphics pipeline
        // ═══════════════════════════════════════════════════════════
        vk::GraphicsPipelineCreateInfo pipelineInfo(
            {},                          // flags
            shaderStages,               // stages
            &vertexInputInfo,           // pVertexInputState
            &inputAssembly,             // pInputAssemblyState
            nullptr,                     // pTessellationState
            &viewportState,             // pViewportState
            &rasterizer,                // pRasterizationState
            &multisampling,             // pMultisampleState
            &depthStencilInfo,           // pDepthStencilState (using depth)
            &colorBlending,             // pColorBlendState
            &dynamicState,              // pDynamicState
            *m_PipelineLayout,          // layout
            nullptr,                     // renderPass (using dynamic rendering)
            0,                           // subpass
            nullptr,                     // basePipelineHandle
            -1                           // basePipelineIndex
        );
        pipelineInfo.pNext = &renderingInfo;

        m_GraphicsPipeline = vk::raii::Pipeline(m_Device.value(), nullptr, pipelineInfo);
        LOG_INFO("VulkanRenderer", "Graphics pipeline created successfully");
    }

    void VulkanRenderer::CreateCommandPool()
    {
        vk::CommandPoolCreateInfo poolInfo(
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,  // flags - allow individual buffer reset
            m_GraphicsQueueFamilyIdx                              // queueFamilyIndex
        );

        m_CommandPool = vk::raii::CommandPool(m_Device.value(), poolInfo);
        LOG_INFO("VulkanRenderer", "Command pool created");
    }

    void VulkanRenderer::CreateCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo(
            *m_CommandPool,                      // commandPool
            vk::CommandBufferLevel::ePrimary,   // level - can be submitted directly to queue
            1                                    // commandBufferCount
        );

        auto commandBuffers = vk::raii::CommandBuffers(m_Device.value(), allocInfo);
        m_CommandBuffer = std::move(commandBuffers[0]);
        LOG_INFO("VulkanRenderer", "Command buffer allocated");
    }

    vk::raii::ShaderModule VulkanRenderer::CreateShaderModule(const std::vector<char>& code)
    {
        vk::ShaderModuleCreateInfo createInfo(
            {},                                 // flags
            code.size(),                       // codeSize
            reinterpret_cast<const uint32_t*>(code.data()) // pCode
        );

        vk::raii::ShaderModule shaderModule(m_Device.value(), createInfo);
        return shaderModule;
    }

    void VulkanRenderer::CreateImageView()
    {
        m_SwapchainImageViews.clear();
        
        // Our images will be used as color targets without any mipmapping levels or multiple layers.
        vk::ImageSubresourceRange subresourceRange(
            vk::ImageAspectFlagBits::eColor,  // aspectMask
            0,                                 // baseMipLevel
            1,                                 // levelCount
            0,                                 // baseArrayLayer
            1                                  // layerCount
        );

        /************************************************************************
        Default components mapping:
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        ************************************************************************/
        vk::ImageViewCreateInfo imageViewCreateInfo(
            {},                                // flags
            {},                                // image (will be set per-image)
            vk::ImageViewType::e2D,           // viewType
            m_SwapchainImageFormat.format,    // format
            {},                                // components (default mapping)
            subresourceRange                   // subresourceRange
        );

        for (const auto& swapchainImage : m_SwapchainImages)
        {
            imageViewCreateInfo.image = swapchainImage;
            vk::raii::ImageView imageView(m_Device.value(), imageViewCreateInfo);
            m_SwapchainImageViews.push_back(std::move(imageView));
            // Store or use the imageView as needed
            // For example, you might want to keep them in a member variable
        }
    }

    void VulkanRenderer::CreateSwapChain()
    {
        auto surfaceCapabilities = m_PhysicalDevice->getSurfaceCapabilitiesKHR(*m_Surface);
        m_SwapchainImageFormat = ChooseSwapSurfaceFormat(
            m_PhysicalDevice->getSurfaceFormatsKHR(*m_Surface)
        );
        m_SwapchainPresentMode = ChooseSwapPresentMode(
            m_PhysicalDevice->getSurfacePresentModesKHR(*m_Surface)
        );
        m_SwapchainExtent = ChooseSwapExtent(surfaceCapabilities, m_Window->GetWidth(), m_Window->GetHeight());
        LOG_DEBUG("VulkanRenderer", "\tChosen swapchain extent: {}x{}", m_SwapchainExtent.width, m_SwapchainExtent.height);
        m_DepthBuffer = CreateDepthBuffer();
        m_ProjectionMatrix = glm::perspective(glm::radians(45.0f), static_cast<float>(m_SwapchainExtent.width) / static_cast<float>(m_SwapchainExtent.height), 0.1f, 100.0f);
        m_ProjectionMatrix[1][1] *= -1; // Invert Y for Vulkan's coordinate system

        uint32_t minImageCount = (surfaceCapabilities.maxImageCount > 0) 
            ? std::min(surfaceCapabilities.maxImageCount, std::max(surfaceCapabilities.minImageCount + 1, 2u)) 
            : std::max(surfaceCapabilities.minImageCount + 1, 2u);

        vk::SwapchainCreateInfoKHR swapChainCreateInfo(
            {},                                     // flags
            *m_Surface,                            // surface
            minImageCount,                        // minImageCount
            m_SwapchainImageFormat.format,        // imageFormat
            m_SwapchainImageFormat.colorSpace,    // imageColorSpace
            m_SwapchainExtent,                    // imageExtent
            1,                                    // imageArrayLayers -- always 1 unless you're developing stereoscopic 3D app
            vk::ImageUsageFlagBits::eColorAttachment, // imageUsage
            vk::SharingMode::eExclusive,          // imageSharingMode
            0, nullptr,                           // queueFamilyIndexCount, pQueueFamilyIndices
            surfaceCapabilities.currentTransform, // preTransform
            vk::CompositeAlphaFlagBitsKHR::eOpaque,    // compositeAlpha
            m_SwapchainPresentMode,               // presentMode
            VK_TRUE,                              // clipped
            m_Swapchain ? **m_Swapchain : nullptr // oldSwapchain
        );

        if (m_GraphicsQueueFamilyIdx != m_PresentQueueFamilyIdx)
        {
            uint32_t queueFamilyIndices[] = { m_GraphicsQueueFamilyIdx, m_PresentQueueFamilyIdx };
            swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            swapChainCreateInfo.queueFamilyIndexCount = 2;
            swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
            swapChainCreateInfo.queueFamilyIndexCount = 0; // Optional
            swapChainCreateInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        m_Swapchain = vk::raii::SwapchainKHR(m_Device.value(), swapChainCreateInfo);
        LOG_DEBUG("VulkanRenderer", "\tCreated swapchain with {} images for extent: {}x{}", m_SwapchainImages.size(), m_SwapchainExtent.width, m_SwapchainExtent.height);
        m_SwapchainImages = m_Swapchain->getImages();
        LOG_DEBUG("VulkanRenderer", "\tRetrieved {} swapchain images for extent: {}x{}", m_SwapchainImages.size(), m_SwapchainExtent.width, m_SwapchainExtent.height);
    }

    void VulkanRenderer::RecreateSwapchain()
    {
        // Implementation for recreating the swapchain
        LOG_DEBUG("VulkanRenderer", "Recreating swapchain...");
        // Actual swapchain recreation logic goes here

        auto surfaceCapabilities = m_PhysicalDevice->getSurfaceCapabilitiesKHR(*m_Surface);
        m_SwapchainExtent = ChooseSwapExtent(surfaceCapabilities, m_Window->GetWidth(), m_Window->GetHeight());
        m_DepthBuffer = CreateDepthBuffer();
        m_ProjectionMatrix = glm::perspective(glm::radians(45.0f), static_cast<float>(m_SwapchainExtent.width) / static_cast<float>(m_SwapchainExtent.height), 0.1f, 100.0f);
        m_ProjectionMatrix[1][1] *= -1; // Invert Y for Vulkan's coordinate system
        if (m_SwapchainExtent.width == 0 || m_SwapchainExtent.height == 0)
        {
            LOG_DEBUG("VulkanRenderer", "Swapchain extent is zero, waiting for window to be resized...");
            return; // Exit early if the swapchain extent is zero
        }

        LOG_DEBUG("VulkanRenderer", "\tm_SwapchainExtent recreated: {}x{}", m_SwapchainExtent.width, m_SwapchainExtent.height);
        
        m_Device->waitIdle();
        m_SwapchainImageViews.clear();
        LOG_DEBUG("VulkanRenderer", "\tCleared swapchain image views for new extent: {}x{}", m_SwapchainExtent.width, m_SwapchainExtent.height);
        CreateSwapChain();
        LOG_DEBUG("VulkanRenderer", "\tCreated new swapchain for extent: {}x{}", m_SwapchainExtent.width, m_SwapchainExtent.height);
        CreateImageView();
        LOG_DEBUG("VulkanRenderer", "\tCreated new image views for extent: {}x{}", m_SwapchainExtent.width, m_SwapchainExtent.height);
        CreateRenderFinishedSemaphores();
        LOG_DEBUG("VulkanRenderer", "\tCreated new render finished semaphores for extent: {}x{}", m_SwapchainExtent.width, m_SwapchainExtent.height);
        m_SubOptimal = false;
    }

    void VulkanRenderer::CreateSurface()
    {
        // Create a Vulkan surface using the native window handle
        VkSurfaceKHR surface;
        GLFWwindow* glfwWindow = static_cast<GLFWwindow*>(m_Window->GetNativeHandle());
        if (glfwCreateWindowSurface(static_cast<VkInstance>(**m_Instance), glfwWindow, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface!");
        }
        // Store or use the surface as needed
        m_Surface = vk::raii::SurfaceKHR(m_Instance.value(), surface);
    }

    void VulkanRenderer::CreateLogicalDevice()
    {
        auto indices = FindQueueFamilies(m_PhysicalDevice.value(), m_Surface.value());

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;

        // Query supported Vulkan 1.3 features from the physical device
        auto supportedFeatures = m_PhysicalDevice->getFeatures2<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceDynamicRenderingFeatures
        >();

        auto& supported13 = supportedFeatures.get<vk::PhysicalDeviceVulkan13Features>();

        LOG_DEBUG("VulkanRenderer", "Device supports dynamicRendering: {}",
                  supported13.dynamicRendering ? "yes" : "no");
        LOG_DEBUG("VulkanRenderer", "Device supports synchronization2: {}",
                  supported13.synchronization2 ? "yes" : "no");

        // Set up features we want to enable (chained via pNext)
        vk::PhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.shaderDrawParameters = VK_TRUE;

        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.dynamicRendering = VK_TRUE;
        vulkan13Features.synchronization2 = VK_TRUE;
        vulkan13Features.pNext = &vulkan11Features;

        vk::PhysicalDeviceFeatures2 features2{};
        features2.pNext = &vulkan13Features;

        // create a Device
        float queuePriority = 1.0f;

        for (uint32_t queueFamily : indices.UniqueFamilies())
        {
            vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
                {},
                queueFamily,
                1,
                &queuePriority
            );
            queueCreateInfos.push_back(deviceQueueCreateInfo);
        }

        vk::DeviceCreateInfo deviceCreateInfo(
            {},                                                 // flags
            static_cast<uint32_t>(queueCreateInfos.size()),     // queueCreateInfoCount
            queueCreateInfos.data(),                           // pQueueDeviceCreateInfos
            0, nullptr,                                         // EnabledLayerCount / EnabledLayerNames
            static_cast<uint32_t>(m_EnabledDeviceExtensions.size()),   // enabledExtensionCount
            m_EnabledDeviceExtensions.data(),                          // ppEnabledExtensionNames
            nullptr                                             // pEnabledFeatures (using pNext instead)
        );
        deviceCreateInfo.pNext = &features2;

        m_Device = vk::raii::Device(m_PhysicalDevice.value(), deviceCreateInfo);
        if (!m_Device)
        {
            throw std::runtime_error("Failed to create logical device!");
        }

        m_GraphicsQueue = vk::raii::Queue(m_Device.value(), m_GraphicsQueueFamilyIdx, 0);
        if (!m_GraphicsQueue)
        {
            throw std::runtime_error("Failed to retrieve graphics queue!");
        }

        m_PresentQueue = vk::raii::Queue(m_Device.value(), m_PresentQueueFamilyIdx, 0);
        if (!m_PresentQueue)
        {
            throw std::runtime_error("Failed to retrieve present queue!");
        }
    }

    void VulkanRenderer::PickPhysicalDevice()
    {
        std::vector<vk::raii::PhysicalDevice> devices = m_Instance->enumeratePhysicalDevices();
        if (devices.empty())
        {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        // Separate function to check device suitability
        auto isDeviceSuitable = [&](const vk::raii::PhysicalDevice& device) -> bool
        {
            LOG_INFO("VulkanRenderer", "Evaluating device: {}", device.getProperties().deviceName.data());

            // Check API version
            if (device.getProperties().apiVersion < VK_API_VERSION_1_3)
            {
                LOG_WARN("VulkanRenderer", "  ❌ Does not support Vulkan 1.3");
                return false;
            }
            LOG_INFO("VulkanRenderer", "  ✓ Supports Vulkan 1.3");
            
            // Check for graphics and presentation queue
            auto indices = FindQueueFamilies(device, m_Surface.value());
            if (!indices.isComplete())
            {
                LOG_WARN("VulkanRenderer", "  ❌ Missing required queue families");
                return false;
            }
            // TODO: Don't set members inside a suitability check. It works only because
            // find_if stops at the first passing device, so the chosen device writes last.
            // Have the check return the indices (and the depth format, set the same way
            // below), then assign them once after m_PhysicalDevice = *deviceIter.
            // value_or is unneeded here: isComplete() already guarantees both values.
            m_GraphicsQueueFamilyIdx = indices.graphicsFamily.value_or(UINT32_MAX);
            m_PresentQueueFamilyIdx = indices.presentFamily.value_or(UINT32_MAX);

            LOG_INFO("VulkanRenderer", "  ✓ Graphics queue family found {}", indices.graphicsFamily.value());
            LOG_INFO("VulkanRenderer", "  ✓ Present queue family found {}", indices.presentFamily.value());

            if (indices.graphicsFamily.value() != indices.presentFamily.value())
            {
                LOG_INFO("VulkanRenderer", "  ℹ️  Graphics and Present queues are different");
            }
            else
            {
                LOG_INFO("VulkanRenderer", "  ℹ️  Graphics and Present queues are the same");
            }

            
            // Check extensions
            auto extensions = device.enumerateDeviceExtensionProperties();
            for (const char* requiredExt : m_RequestedDeviceExtensions)
            {
                auto extIter = std::ranges::find_if(extensions, [requiredExt](const auto& ext) {
                    return strcmp(ext.extensionName, requiredExt) == 0;
                });

                if (extIter == extensions.end())
                {
                    LOG_WARN("VulkanRenderer", "  ❌ Missing extension: {}", requiredExt);
                    return false;
                }
                LOG_INFO("VulkanRenderer", "  ✓ Supports {}", requiredExt);
            }

            // Check feature11
            auto candidate = device.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features>();
            const auto& feature11 = candidate.get<vk::PhysicalDeviceVulkan11Features>();
            if (!feature11.shaderDrawParameters)
            {
                LOG_WARN("VulkanRenderer", "  ❌ Does not support shaderDrawParameters");
                return false;
            }
            LOG_INFO("VulkanRenderer", "  ✓ Supports shaderDrawParameters");

            // Select a suitable depth format
            std::vector<vk::Format> depthFormats = {
                vk::Format::eD32Sfloat,
                // vk::Format::eD32SfloatS8Uint,
                // vk::Format::eD24UnormS8Uint
            };
            for (auto format : depthFormats)
            {
                vk::FormatProperties props = device.getFormatProperties(format);
                if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
                {
                    m_DepthFormat = format;
                    break;
                }
            }
            if (m_DepthFormat == vk::Format{})
            {
                LOG_WARN("VulkanRenderer", "  ❌ No suitable depth format found");
                return false;
            }
            LOG_INFO("VulkanRenderer", "  ✓ Selected depth format: {}", vk::to_string(m_DepthFormat));
            
            return true;
        };
        
        // Find first suitable device
        auto deviceIter = std::ranges::find_if(devices, isDeviceSuitable);
        
        if (deviceIter == devices.end())
        {
            throw std::runtime_error("Failed to find suitable GPU");
        }
        
        // Only NOW do we assign and populate enabled extensions
        m_PhysicalDevice = *deviceIter;
        m_EnabledDeviceExtensions = m_RequestedDeviceExtensions;  // All were validated

        // Check and add optional extensions if available
        auto availableExtensions = m_PhysicalDevice->enumerateDeviceExtensionProperties();
        for (const char* optionalExt : m_OptionalDeviceExtensions)
        {
            auto extIter = std::ranges::find_if(availableExtensions, [optionalExt](const auto& ext) {
                return strcmp(ext.extensionName, optionalExt) == 0;
            });

            if (extIter != availableExtensions.end())
            {
                LOG_INFO("VulkanRenderer", "  ✓ Enabling optional extension: {}", optionalExt);
                m_EnabledDeviceExtensions.push_back(optionalExt);
            }
            else
            {
                LOG_INFO("VulkanRenderer", "  ℹ️  Optional extension not available: {}", optionalExt);
            }
        }

        LOG_INFO("VulkanRenderer", "✅ Selected device: {}", m_PhysicalDevice->getProperties().deviceName.data());
    }


    void VulkanRenderer::CreateInstance()
    {
        LOG_INFO("VulkanRenderer", "Creating Vulkan instance...");
        vk::ApplicationInfo appInfo(
            "KHClone",                      // pApplicationName
            VK_MAKE_VERSION(1, 0, 0),     // applicationVersion
            "Momo Engine",                   // pEngineName
            VK_MAKE_VERSION(1, 0, 0),     // engineVersion
            vk::ApiVersion14              // apiVersion
        );

        // Get required extensions from GLFW
        std::vector<const char*> extensions;
        Momo::WindowVulkan::GetRequiredVulkanExtensions(*m_Window, extensions);

        // Check if the required GLFW extensions are supported by Vulkan implementation
        auto extensionProperties = m_Context.enumerateInstanceExtensionProperties();
        for (const char* requiredExt : extensions)
        {
            if (std::ranges::none_of(extensionProperties,
                [&](const vk::ExtensionProperties& prop) {
                    return std::strcmp(prop.extensionName, requiredExt) == 0;
                }))
            {
                throw std::runtime_error("Required GLFW Vulkan extension not supported: " + std::string(requiredExt));
            }
        }

        // Add portability enumeration extension for MoltenVK on macOS
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

        vk::InstanceCreateInfo instanceCreateInfo;
        std::array<vk::ValidationFeatureEnableEXT, 1> validationFeaturesArray {
            vk::ValidationFeatureEnableEXT::eSynchronizationValidation
        };

        vk::ValidationFeaturesEXT validationFeatures = vk::ValidationFeaturesEXT(
            static_cast<uint32_t>(validationFeaturesArray.size()),
            validationFeaturesArray.data(),
            0,
            nullptr,
            nullptr
        );

        // Add debug utils extension on Debug builds
        if (enableValidationLayers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            // Check if m_ValidationLayers is available via enumerateInstanceLayerProperties
            auto layerProperties = m_Context.enumerateInstanceLayerProperties();
            for (const char* layerName : m_ValidationLayers)
            {
                if (std::ranges::none_of(layerProperties,
                    [&](const vk::LayerProperties& prop) {
                        return std::strcmp(prop.layerName, layerName) == 0;
                    }))
                {
                    throw std::runtime_error("Required validation layer not supported: " + std::string(layerName));
                }
            }

            instanceCreateInfo = vk::InstanceCreateInfo(
                vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,  // flags - required for MoltenVK
                &appInfo,                                     // pApplicationInfo
                static_cast<uint32_t>(m_ValidationLayers.size()), 
                m_ValidationLayers.data(),                                   // enabled layers (count, names)
                static_cast<uint32_t>(extensions.size()),    // enabled extensions count
                extensions.data(),                             // enabled extension names
                &validationFeatures
            );
        }
        else
        {
            instanceCreateInfo = vk::InstanceCreateInfo(
                vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR,  // flags - required for MoltenVK
                &appInfo,                                     // pApplicationInfo
                0, 
                nullptr,                                   // enabled layers (count, names)
                static_cast<uint32_t>(extensions.size()),    // enabled extensions count
                extensions.data()                             // enabled extension names
            );
        }

        // Create the Vulkan instance
        m_Instance = vk::raii::Instance(m_Context, instanceCreateInfo);
    }

    void VulkanRenderer::CreateSyncObjects()
    {
        CreateImageAvailableSemaphore();
        CreateInFlightFence();
        CreateRenderFinishedSemaphores();
    }

    void VulkanRenderer::CreateRenderFinishedSemaphores()
    {
        m_RenderFinishedSemaphores.clear();
        vk::SemaphoreCreateInfo semaphoreInfo{};
        for (auto swapchain : m_SwapchainImages)
            m_RenderFinishedSemaphores.push_back(vk::raii::Semaphore(m_Device.value(), semaphoreInfo));
    }

    void VulkanRenderer::CreateImageAvailableSemaphore()
    {
        vk::SemaphoreCreateInfo semaphoreInfo{};
        m_ImageAvailableSemaphore = vk::raii::Semaphore(m_Device.value(), semaphoreInfo);
    }

    void VulkanRenderer::CreateInFlightFence()
    {
        vk::FenceCreateInfo fenceInfo{};
        fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
        m_InFlightFence = vk::raii::Fence(m_Device.value(), fenceInfo);
    }
} // namespace Renderer
} // namespace Momo
