// Screen-space ambient occlusion for the raster modes: a depth prepass from
// the camera (prepass.slang), then ssao.slang's occlusion and blur passes. The
// raster pass reads the result (lighting.slang, set 1 binding 4) to dim the sky
// light, and tests against the prepass's depth instead of clearing it.
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Logging/Logger.h"

namespace Momo {
namespace Renderer {
    namespace
    {
        constexpr uint32_t OcclusionGroupSize = 8; // ssao.slang's numthreads
    }

    void VulkanRenderer::CreateOcclusionPipelines()
    {
        // Depth prepass: the raster pass's vertex stage and alpha test, no color
        auto vertShaderModule = CreateShaderModule(ReadFile("shaders/prepass.vert.spv"));
        auto fragShaderModule = CreateShaderModule(ReadFile("shaders/prepass.frag.spv"));
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "vertMain"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "fragMain")
        };
        vk::VertexInputBindingDescription binding(0, sizeof(Geometry::Vertex), vk::VertexInputRate::eVertex);
        std::array<vk::VertexInputAttributeDescription, 3> attributes{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Geometry::Vertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Geometry::Vertex, normal)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Geometry::Vertex, texCoord)),
        };
        vk::PipelineVertexInputStateCreateInfo vertexInput({}, binding, attributes);
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
        // Culls exactly as the raster pass does (per mesh, dynamic), or a back
        // face it skips would hide what it draws behind
        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.lineWidth = 1.0f;
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1);
        vk::PipelineDepthStencilStateCreateInfo depthStencil({}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess);
        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        std::array<vk::DynamicState, 3> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eCullMode };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);
        vk::PipelineRenderingCreateInfo renderingInfo({}, 0, nullptr, m_DepthFormat);

        vk::GraphicsPipelineCreateInfo pipelineInfo(
            {}, shaderStages, &vertexInput, &inputAssembly, nullptr, &viewportState, &rasterizer,
            &multisampling, &depthStencil, &colorBlending, &dynamicState, **m_PipelineLayout);
        pipelineInfo.pNext = &renderingInfo;
        m_DepthPrepassPipeline = vk::raii::Pipeline(*m_Device, nullptr, pipelineInfo);

        std::array<vk::DescriptorSetLayout, 2> setLayouts{ **m_OcclusionSetLayout, **m_FrameSetLayout };
        m_OcclusionPipelineLayout = vk::raii::PipelineLayout(*m_Device, vk::PipelineLayoutCreateInfo({}, setLayouts));
        auto createComputePipeline = [this](const std::string& shaderPath) {
            vk::raii::ShaderModule shaderModule = CreateShaderModule(ReadFile(shaderPath));
            vk::ComputePipelineCreateInfo computeInfo(
                {},
                vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "compMain"),
                **m_OcclusionPipelineLayout
            );
            return vk::raii::Pipeline(*m_Device, nullptr, computeInfo);
        };
        m_OcclusionPipeline = createComputePipeline("shaders/ssao.comp.spv");
        m_OcclusionBlurPipeline = createComputePipeline("shaders/ssao_blur.comp.spv");
    }

    // Called whenever the swapchain (and so the depth buffer) is recreated
    void VulkanRenderer::CreateOcclusionImages()
    {
        m_RawOcclusion.emplace(CreateRayTracingStorageImage(OcclusionFormat, {}));
        m_Occlusion.emplace(CreateRayTracingStorageImage(OcclusionFormat, vk::ImageUsageFlagBits::eSampled));

        // General from here on: written as storage, read by the raster pass
        auto commandBuffer = BeginSingleUseCommandBuffer();
        std::array<vk::ImageMemoryBarrier2, 2> toGeneral{};
        for (size_t i = 0; i < toGeneral.size(); i++) {
            toGeneral[i].srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
            toGeneral[i].dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            toGeneral[i].oldLayout = vk::ImageLayout::eUndefined;
            toGeneral[i].newLayout = vk::ImageLayout::eGeneral;
            toGeneral[i].subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
        }
        toGeneral[0].image = *m_RawOcclusion->image;
        toGeneral[1].image = *m_Occlusion->image;
        vk::DependencyInfo dependency{};
        dependency.setImageMemoryBarriers(toGeneral);
        commandBuffer.pipelineBarrier2(dependency);
        EndSingleUseCommandBuffer(commandBuffer);

        vk::DescriptorImageInfo depthInfo(**m_NearestSampler, *m_DepthBuffer->imageView, vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo rawInfo({}, *m_RawOcclusion->imageView, vk::ImageLayout::eGeneral);
        vk::DescriptorImageInfo occlusionStorageInfo({}, *m_Occlusion->imageView, vk::ImageLayout::eGeneral);
        vk::DescriptorImageInfo occlusionInfo(**m_NearestSampler, *m_Occlusion->imageView, vk::ImageLayout::eGeneral);
        std::vector<vk::WriteDescriptorSet> writes{
            vk::WriteDescriptorSet(m_OcclusionDescriptorSet, 0, 0, vk::DescriptorType::eCombinedImageSampler, depthInfo),
            vk::WriteDescriptorSet(m_OcclusionDescriptorSet, 1, 0, vk::DescriptorType::eStorageImage, rawInfo),
            vk::WriteDescriptorSet(m_OcclusionDescriptorSet, 2, 0, vk::DescriptorType::eStorageImage, occlusionStorageInfo),
        };
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            writes.emplace_back(m_FrameDescriptorSets[i], 4, 0, vk::DescriptorType::eCombinedImageSampler, occlusionInfo);
        }
        m_Device->updateDescriptorSets(writes, nullptr);
    }

    void VulkanRenderer::RecordDepthPrepass(const VulkanModelData& modelData, const glm::mat4& viewMatrix, const glm::mat4& modelMatrix)
    {
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        // The depth buffer is shared by all frames in flight: wait for the
        // previous frame's depth writes and its composite pass's reads
        vk::ImageMemoryBarrier2 toAttachment{};
        toAttachment.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests
                                  | vk::PipelineStageFlagBits2::eFragmentShader;
        toAttachment.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        toAttachment.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        toAttachment.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        toAttachment.oldLayout = vk::ImageLayout::eUndefined;
        toAttachment.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        toAttachment.image = *m_DepthBuffer->image;
        toAttachment.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
        vk::DependencyInfo dependency{};
        dependency.setImageMemoryBarriers(toAttachment);
        commandBuffer.pipelineBarrier2(dependency);

        vk::RenderingAttachmentInfo depthAttachment{};
        depthAttachment.imageView = *m_DepthBuffer->imageView;
        depthAttachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachment.clearValue = vk::ClearValue().setDepthStencil({ 1.0f, 0 });
        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D({ 0, 0 }, m_SwapchainExtent);
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;

        commandBuffer.beginRendering(renderingInfo);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **m_DepthPrepassPipeline);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(m_SwapchainExtent.width), static_cast<float>(m_SwapchainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, m_SwapchainExtent));
        for (const auto& mesh : modelData.meshes) {
            commandBuffer.setCullMode(mesh.materialData.doubleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **m_PipelineLayout, 0, { mesh.materialData.descriptorSet }, nullptr);
            commandBuffer.bindVertexBuffers(0, *mesh.gpuMesh.vertexBuffer.buffer, { 0 });
            commandBuffer.bindIndexBuffer(*mesh.gpuMesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);
            // The same matrices as the raster pass, for the same depths
            PushConstantData pushConstantData{
                .projectionMatrix = m_ProjectionMatrix,
                .viewMatrix = viewMatrix,
                .modelMatrix = modelMatrix * mesh.localTransform,
            };
            commandBuffer.pushConstants<PushConstantData>(**m_PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstantData);
            commandBuffer.drawIndexed(mesh.gpuMesh.indexBuffer.indexCount, 1, 0, 0, 0);
        }
        commandBuffer.endRendering();
    }

    void VulkanRenderer::RecordOcclusionPass()
    {
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        const vk::ImageSubresourceRange depthRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

        // Prepass depth -> sampled here
        vk::ImageMemoryBarrier2 depthToRead{};
        depthToRead.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        depthToRead.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        depthToRead.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        depthToRead.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
        depthToRead.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        depthToRead.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        depthToRead.image = *m_DepthBuffer->image;
        depthToRead.subresourceRange = depthRange;
        // The occlusion images are shared by all frames in flight: the
        // previous frame's blur and raster pass must be done reading them
        vk::MemoryBarrier2 previousReads(
            vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eFragmentShader, {},
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
        vk::DependencyInfo before{};
        before.setMemoryBarriers(previousReads);
        before.setImageMemoryBarriers(depthToRead);
        commandBuffer.pipelineBarrier2(before);

        std::array<vk::DescriptorSet, 2> descriptorSets{ m_OcclusionDescriptorSet, m_FrameDescriptorSets[m_CurrentFrame] };
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **m_OcclusionPipelineLayout, 0, descriptorSets, nullptr);
        uint32_t groupsX = (m_SwapchainExtent.width + OcclusionGroupSize - 1) / OcclusionGroupSize;
        uint32_t groupsY = (m_SwapchainExtent.height + OcclusionGroupSize - 1) / OcclusionGroupSize;
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **m_OcclusionPipeline);
        commandBuffer.dispatch(groupsX, groupsY, 1);

        // Raw occlusion -> the blur
        vk::MemoryBarrier2 rawToBlur(
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead);
        vk::DependencyInfo between{};
        between.setMemoryBarriers(rawToBlur);
        commandBuffer.pipelineBarrier2(between);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **m_OcclusionBlurPipeline);
        commandBuffer.dispatch(groupsX, groupsY, 1);

        // Occlusion -> the raster pass; depth back to an attachment for it
        vk::MemoryBarrier2 occlusionToRead(
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
            vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead);
        vk::ImageMemoryBarrier2 depthToAttachment{};
        depthToAttachment.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        depthToAttachment.srcAccessMask = {};
        depthToAttachment.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        depthToAttachment.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        depthToAttachment.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        depthToAttachment.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        depthToAttachment.image = *m_DepthBuffer->image;
        depthToAttachment.subresourceRange = depthRange;
        vk::DependencyInfo after{};
        after.setMemoryBarriers(occlusionToRead);
        after.setImageMemoryBarriers(depthToAttachment);
        commandBuffer.pipelineBarrier2(after);
    }
} // namespace Renderer
} // namespace Momo
