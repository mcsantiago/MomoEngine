// Lighting shared by every pipeline (lighting.slang, descriptor set 1) and the
// raster path's composite pass (composite.slang)
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Logging/Logger.h"

namespace Momo {
namespace Renderer {
    void VulkanRenderer::CreateFrameResources()
    {
        // Binding 0: FrameUniforms. 1: point lights. 2: the sun's shadow map.
        // 3: point lights' cube shadow maps. 4: ambient occlusion.
        constexpr vk::ShaderStageFlags stages = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute;
        std::array<vk::DescriptorSetLayoutBinding, 5> bindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, stages),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, stages),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1, stages),
            vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eCombinedImageSampler, MaxShadowedPointLights, stages),
            vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eCombinedImageSampler, 1, stages),
        };
        m_FrameSetLayout = vk::raii::DescriptorSetLayout(*m_Device, vk::DescriptorSetLayoutCreateInfo({}, bindings));

        // Composite pass: depth and normals of the raster pass
        std::array<vk::DescriptorSetLayoutBinding, 2> compositeBindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
        };
        m_CompositeSetLayout = vk::raii::DescriptorSetLayout(*m_Device, vk::DescriptorSetLayoutCreateInfo({}, compositeBindings));

        // Ambient occlusion: the prepass's depth, and the occlusion before and after the blur
        std::array<vk::DescriptorSetLayoutBinding, 3> occlusionBindings{
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
            vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute),
        };
        m_OcclusionSetLayout = vk::raii::DescriptorSetLayout(*m_Device, vk::DescriptorSetLayoutCreateInfo({}, occlusionBindings));

        m_FrameDescriptorAllocator.Init(*m_Device, DescriptorPoolSizes{
            .maxSets = MAX_FRAMES_IN_FLIGHT + 2,
            .sizes = {
                { vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT },
                { vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT },
                { vk::DescriptorType::eCombinedImageSampler, 3 + MAX_FRAMES_IN_FLIGHT * (2 + MaxShadowedPointLights) },
                { vk::DescriptorType::eStorageImage, 2 },
            },
        });

        FrameUniforms initialUniforms{};
        std::array<GpuPointLight, MaxPointLights> initialLights{};
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            m_FrameUniformBuffers.push_back(CreateBuffer(&initialUniforms, sizeof(FrameUniforms), vk::BufferUsageFlagBits::eUniformBuffer, 0));
            m_FrameUniformsMapped[i] = m_FrameUniformBuffers[i].memory.mapMemory(0, sizeof(FrameUniforms));
            m_PointLightBuffers.push_back(CreateBuffer(initialLights.data(), sizeof(initialLights), vk::BufferUsageFlagBits::eStorageBuffer, 0));
            m_PointLightsMapped[i] = m_PointLightBuffers[i].memory.mapMemory(0, sizeof(initialLights));
            m_FrameDescriptorSets[i] = m_FrameDescriptorAllocator.AllocateDescriptorSet(**m_FrameSetLayout);
        }
        m_CompositeDescriptorSet = m_FrameDescriptorAllocator.AllocateDescriptorSet(**m_CompositeSetLayout);
        m_OcclusionDescriptorSet = m_FrameDescriptorAllocator.AllocateDescriptorSet(**m_OcclusionSetLayout);

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eNearest;
        samplerInfo.minFilter = vk::Filter::eNearest;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        m_NearestSampler = vk::raii::Sampler(*m_Device, samplerInfo);

        CreateShadowMap();
        CreatePointShadowMaps();
        std::vector<vk::DescriptorImageInfo> pointShadowInfos;
        for (const PointShadowMap& map : m_PointShadowMaps) {
            pointShadowInfos.emplace_back(**m_ShadowSampler, *map.cube.imageView, vk::ImageLayout::eShaderReadOnlyOptimal);
        }

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::DescriptorBufferInfo uniformInfo(*m_FrameUniformBuffers[i].buffer, 0, sizeof(FrameUniforms));
            vk::DescriptorBufferInfo lightInfo(*m_PointLightBuffers[i].buffer, 0, vk::WholeSize);
            vk::DescriptorImageInfo shadowInfo(**m_ShadowSampler, *m_ShadowMap->imageView, vk::ImageLayout::eShaderReadOnlyOptimal);
            std::array<vk::WriteDescriptorSet, 4> writes{
                vk::WriteDescriptorSet(m_FrameDescriptorSets[i], 0, 0, vk::DescriptorType::eUniformBuffer, nullptr, uniformInfo),
                vk::WriteDescriptorSet(m_FrameDescriptorSets[i], 1, 0, vk::DescriptorType::eStorageBuffer, nullptr, lightInfo),
                vk::WriteDescriptorSet(m_FrameDescriptorSets[i], 2, 0, vk::DescriptorType::eCombinedImageSampler, shadowInfo),
                vk::WriteDescriptorSet(m_FrameDescriptorSets[i], 3, 0, vk::DescriptorType::eCombinedImageSampler, pointShadowInfos),
            };
            m_Device->updateDescriptorSets(writes, nullptr);
        }
    }

    void VulkanRenderer::SetPointLights(const std::vector<Assets::PointLight>& lights)
    {
        if (lights.size() > MaxPointLights) {
            LOG_WARN("VulkanRenderer", "Scene has {} point lights; using the first {}", lights.size(), MaxPointLights);
        }
        std::vector<GpuPointLight> gpuLights;
        for (size_t i = 0; i < std::min<size_t>(lights.size(), MaxPointLights); i++) {
            gpuLights.push_back(GpuPointLight{
                .position = glm::vec4(lights[i].position, 1.0f),
                .intensity = glm::vec4(lights[i].intensity, 0.0f),
            });
        }
        // The path tracer's running average was made with the old lights
        if (gpuLights != m_PointLights) {
            m_PointLights = std::move(gpuLights);
            m_AccumulatedSamples = 0;
        }
    }

    // The current frame slot's fence has been waited on, so the GPU is done with its buffer
    void VulkanRenderer::WriteFrameUniforms(const glm::mat4& viewMatrix, RenderMode mode)
    {
        const SceneLighting& lighting = m_Lighting;
        const ToonSettings& toon = m_Toon;
        FrameUniforms uniforms{
            .inverseViewProjection = glm::inverse(m_ProjectionMatrix * viewMatrix),
            .sunViewProjection = SunViewProjection(),
            .viewProjection = m_ProjectionMatrix * viewMatrix,
            .cameraPosition = glm::vec4(glm::vec3(glm::inverse(viewMatrix)[3]), 1.0f),
            .sunDirection = glm::vec4(lighting.SunDirection(), 0.0f),
            .sunIrradiance = glm::vec4(lighting.sunColor * lighting.sunIntensity, 0.0f),
            .skyGround = glm::vec4(lighting.skyGround, 0.0f),
            .skyHorizon = glm::vec4(lighting.skyHorizon, 0.0f),
            .skyZenith = glm::vec4(lighting.skyZenith, 0.0f),
            .exposure = lighting.exposure,
            .pointLightCount = static_cast<uint32_t>(m_PointLights.size()),
            .shadingModel = static_cast<uint32_t>(mode == RenderMode::Toon ? ShadingModel::Toon : ShadingModel::Lambert),
            .nearPlane = NearPlane,
            .farPlane = FarPlane,
            // The path tracer traces its own shadows and never reads the map
            .shadowsEnabled = lighting.shadows && mode != RenderMode::RayTraced ? 1u : 0u,
            .shadowNormalOffset = lighting.shadowNormalBias * ShadowTexelWorldSize(),
            .shadowFilterRadius = lighting.shadowSoftness,
            .pointShadowCount = mode == RenderMode::RayTraced ? 0u : PointShadowCount(),
            .pointShadowNear = PointShadowNear,
            .pointShadowFar = m_PointShadowFar,
            .pointShadowBias = lighting.pointShadowBias,
            .occlusionEnabled = lighting.ambientOcclusion && mode != RenderMode::RayTraced ? 1u : 0u,
            .occlusionRadius = lighting.occlusionRadius,
            .occlusionStrength = lighting.occlusionStrength,
            .showOcclusion = lighting.showOcclusion && mode != RenderMode::RayTraced ? 1u : 0u,
            .toon = ToonUniforms{
                .shadowTint = glm::vec4(toon.shadowTint, 0.0f),
                .rimColor = glm::vec4(toon.rimColor, 0.0f),
                .outlineColor = glm::vec4(toon.outlineColor, 0.0f),
                .bands = static_cast<uint32_t>(std::max(toon.bands, 2)),
                .bandSoftness = toon.bandSoftness,
                .specularStrength = toon.specularStrength,
                .specularSize = toon.specularSize,
                .rimStrength = toon.rimStrength,
                .rimWidth = toon.rimWidth,
                .outlineWidth = toon.outlineWidth,
                .depthThreshold = toon.depthThreshold,
                .normalThreshold = toon.normalThreshold,
                .padding = {},
            },
        };
        std::memcpy(m_FrameUniformsMapped[m_CurrentFrame], &uniforms, sizeof(uniforms));
        if (!m_PointLights.empty()) {
            std::memcpy(m_PointLightsMapped[m_CurrentFrame], m_PointLights.data(), sizeof(GpuPointLight) * m_PointLights.size());
        }
    }

    AllocatedImage VulkanRenderer::CreateNormalBuffer()
    {
        auto image = vk::raii::Image(*m_Device, vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            NormalBufferFormat,
            vk::Extent3D{ m_SwapchainExtent.width, m_SwapchainExtent.height, 1 },
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
            vk::SharingMode::eExclusive,
            0,
            nullptr,
            vk::ImageLayout::eUndefined
        ));
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto imageMemory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex));
        image.bindMemory(*imageMemory, 0);

        vk::ImageViewCreateInfo viewInfo(
            {},
            *image,
            vk::ImageViewType::e2D,
            NormalBufferFormat,
            {},
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );
        auto imageView = vk::raii::ImageView(*m_Device, viewInfo);

        return AllocatedImage{ std::move(imageMemory), std::move(image), std::move(imageView), m_SwapchainExtent.width, m_SwapchainExtent.height };
    }

    // Called whenever the depth and normal buffers are recreated
    void VulkanRenderer::WriteCompositeDescriptorSet()
    {
        vk::DescriptorImageInfo depthInfo(**m_NearestSampler, *m_DepthBuffer->imageView, vk::ImageLayout::eShaderReadOnlyOptimal);
        vk::DescriptorImageInfo normalInfo(**m_NearestSampler, *m_NormalBuffer->imageView, vk::ImageLayout::eShaderReadOnlyOptimal);
        std::array<vk::WriteDescriptorSet, 2> writes{
            vk::WriteDescriptorSet(m_CompositeDescriptorSet, 0, 0, vk::DescriptorType::eCombinedImageSampler, depthInfo),
            vk::WriteDescriptorSet(m_CompositeDescriptorSet, 1, 0, vk::DescriptorType::eCombinedImageSampler, normalInfo),
        };
        m_Device->updateDescriptorSets(writes, nullptr);
    }

    void VulkanRenderer::CreateCompositePipeline()
    {
        auto vertShaderModule = CreateShaderModule(ReadFile("shaders/composite.vert.spv"));
        auto fragShaderModule = CreateShaderModule(ReadFile("shaders/composite.frag.spv"));
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "vertMain"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "fragMain")
        };

        // The full-screen triangle comes from the vertex index alone
        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.cullMode = vk::CullModeFlagBits::eNone;
        rasterizer.lineWidth = 1.0f;
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1);

        // Outlines can be partly transparent; the sky is opaque
        vk::PipelineColorBlendAttachmentState blend(
            VK_TRUE,
            vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
            vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        );
        vk::PipelineColorBlendStateCreateInfo colorBlending({}, VK_FALSE, vk::LogicOp::eCopy, 1, &blend);

        std::array<vk::DynamicState, 2> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

        std::array<vk::DescriptorSetLayout, 2> setLayouts{ **m_CompositeSetLayout, **m_FrameSetLayout };
        m_CompositePipelineLayout = vk::raii::PipelineLayout(*m_Device, vk::PipelineLayoutCreateInfo({}, setLayouts));

        vk::Format colorFormat = m_SwapchainImageFormat.format;
        vk::PipelineRenderingCreateInfo renderingInfo({}, 1, &colorFormat);

        vk::GraphicsPipelineCreateInfo pipelineInfo(
            {}, shaderStages, &vertexInput, &inputAssembly, nullptr, &viewportState, &rasterizer,
            &multisampling, nullptr, &colorBlending, &dynamicState, *m_CompositePipelineLayout);
        pipelineInfo.pNext = &renderingInfo;
        m_CompositePipeline = vk::raii::Pipeline(*m_Device, nullptr, pipelineInfo);
    }

    void VulkanRenderer::RecordCompositePass(uint32_t imageIndex)
    {
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        const vk::ImageSubresourceRange colorRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

        // The raster pass's depth and normal writes -> reads here
        vk::ImageMemoryBarrier2 depthToRead{};
        depthToRead.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        depthToRead.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        depthToRead.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        depthToRead.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
        depthToRead.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        depthToRead.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        depthToRead.image = *m_DepthBuffer->image;
        depthToRead.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

        vk::ImageMemoryBarrier2 normalsToRead{};
        normalsToRead.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        normalsToRead.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        normalsToRead.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        normalsToRead.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
        normalsToRead.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        normalsToRead.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        normalsToRead.image = *m_NormalBuffer->image;
        normalsToRead.subresourceRange = colorRange;

        std::array<vk::ImageMemoryBarrier2, 2> barriers{ depthToRead, normalsToRead };
        vk::DependencyInfo dependency{};
        dependency.setImageMemoryBarriers(barriers);
        commandBuffer.pipelineBarrier2(dependency);

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
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **m_CompositePipeline);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(m_SwapchainExtent.width), static_cast<float>(m_SwapchainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, m_SwapchainExtent));
        std::array<vk::DescriptorSet, 2> descriptorSets{ m_CompositeDescriptorSet, m_FrameDescriptorSets[m_CurrentFrame] };
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **m_CompositePipelineLayout, 0, descriptorSets, nullptr);
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRendering();
    }
} // namespace Renderer
} // namespace Momo
