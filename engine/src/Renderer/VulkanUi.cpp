// Dear ImGui drawn on top of every render mode
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Logging/Logger.h"
#include <imgui.h>
#include <imgui_impl_vulkan.h>

namespace Momo {
namespace Renderer {
    void VulkanRenderer::InitImGui()
    {
        m_ImGuiColorFormat = m_SwapchainImageFormat.format;

        ImGui_ImplVulkan_InitInfo info{};
        info.ApiVersion = VK_API_VERSION_1_3;
        info.Instance = **m_Instance;
        info.PhysicalDevice = **m_PhysicalDevice;
        info.Device = **m_Device;
        info.QueueFamily = m_GraphicsQueueFamilyIdx;
        info.Queue = **m_GraphicsQueue;
        // The backend makes its own pool, sized for the font atlas
        info.DescriptorPoolSize = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE;
        info.MinImageCount = MAX_FRAMES_IN_FLIGHT;
        info.ImageCount = static_cast<uint32_t>(std::max<size_t>(m_SwapchainImages.size(), MAX_FRAMES_IN_FLIGHT));
        info.UseDynamicRendering = true;
        info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
        info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = reinterpret_cast<const VkFormat*>(&m_ImGuiColorFormat);
        if (!ImGui_ImplVulkan_Init(&info)) {
            LOG_ERROR("VulkanRenderer", "Failed to initialize the ImGui Vulkan backend");
            return;
        }
        m_ImGuiInitialized = true;
    }

    bool VulkanRenderer::RecordUiPass(uint32_t imageIndex, RenderMode mode)
    {
        ImDrawData* drawData = m_ImGuiInitialized ? ImGui::GetDrawData() : nullptr;
        if (!drawData || drawData->CmdListsCount == 0) {
            return false;
        }
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];

        // The ray tracing pass leaves the image as a blit target
        if (mode == RenderMode::RayTraced) {
            vk::ImageMemoryBarrier2 toColorAttachment{};
            toColorAttachment.srcStageMask = vk::PipelineStageFlagBits2::eBlit;
            toColorAttachment.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
            toColorAttachment.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
            toColorAttachment.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
            toColorAttachment.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            toColorAttachment.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
            toColorAttachment.image = m_SwapchainImages[imageIndex];
            toColorAttachment.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
            vk::DependencyInfo dependency{};
            dependency.setImageMemoryBarriers(toColorAttachment);
            commandBuffer.pipelineBarrier2(dependency);
        }

        vk::RenderingAttachmentInfo colorAttachment{};
        colorAttachment.imageView = *m_SwapchainImageViews[imageIndex];
        colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eLoad;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;

        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D({ 0, 0 }, m_SwapchainExtent);
        renderingInfo.layerCount = 1;
        renderingInfo.setColorAttachments(colorAttachment);

        commandBuffer.beginRendering(renderingInfo);
        ImGui_ImplVulkan_RenderDrawData(drawData, *commandBuffer);
        commandBuffer.endRendering();
        return true;
    }
} // namespace Renderer
} // namespace Momo
