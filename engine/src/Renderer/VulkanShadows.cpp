// Sun shadow map for the raster pass (shadow.slang writes it, lighting.slang reads it)
#include "Momo/Renderer/VulkanRenderer.h"
#include <glm/gtc/matrix_transform.hpp>

namespace Momo {
namespace Renderer {
    void VulkanRenderer::SetShadowBounds(const glm::vec3& min, const glm::vec3& max)
    {
        m_ShadowBoundsMin = min;
        m_ShadowBoundsMax = max;
        // From anywhere in the bounds, the far corner is at most the diagonal away
        m_PointShadowFar = std::max(glm::length(max - min), 1.0f);
        m_PointShadowPositions.clear();
    }

    glm::mat4 VulkanRenderer::SunViewProjection() const
    {
        // A box around the bounds' enclosing sphere fits them from any sun direction
        glm::vec3 center = (m_ShadowBoundsMin + m_ShadowBoundsMax) * 0.5f;
        float radius = std::max(glm::length(m_ShadowBoundsMax - m_ShadowBoundsMin) * 0.5f, 0.01f);
        glm::vec3 toSun = m_Lighting.SunDirection();
        glm::vec3 up = std::abs(toSun.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::mat4 view = glm::lookAt(center + toSun * radius, center, up);
        glm::mat4 projection = glm::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * radius);
        return projection * view;
    }

    float VulkanRenderer::ShadowTexelWorldSize() const
    {
        return glm::length(m_ShadowBoundsMax - m_ShadowBoundsMin) / static_cast<float>(ShadowMapSize);
    }

    void VulkanRenderer::CreateShadowMap()
    {
        auto image = vk::raii::Image(*m_Device, vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            m_DepthFormat,
            vk::Extent3D{ ShadowMapSize, ShadowMapSize, 1 },
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled,
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
            {}, *image, vk::ImageViewType::e2D, m_DepthFormat, {},
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));
        auto imageView = vk::raii::ImageView(*m_Device, viewInfo);
        m_ShadowMap.emplace(AllocatedImage{ std::move(imageMemory), std::move(image), std::move(imageView), ShadowMapSize, ShadowMapSize });

        // Every frame's descriptor set points at the map, and shaders that
        // declare it expect it readable even in frames without a shadow pass
        // (shadows off), so it starts out readable
        auto commandBuffer = BeginSingleUseCommandBuffer();
        vk::ImageMemoryBarrier2 toReadable{};
        toReadable.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        toReadable.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        toReadable.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
        toReadable.oldLayout = vk::ImageLayout::eUndefined;
        toReadable.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        toReadable.image = *m_ShadowMap->image;
        toReadable.subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
        vk::DependencyInfo dependency{};
        dependency.setImageMemoryBarriers(toReadable);
        commandBuffer.pipelineBarrier2(dependency);
        EndSingleUseCommandBuffer(commandBuffer);

        // Comparison sampler: each fetch returns how much of the bilinear
        // footprint passes the depth test, which softens edges for free.
        // Outside the map counts as lit.
        bool linear = static_cast<bool>(m_PhysicalDevice->getFormatProperties(m_DepthFormat).optimalTilingFeatures
                                        & vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = linear ? vk::Filter::eLinear : vk::Filter::eNearest;
        samplerInfo.minFilter = samplerInfo.magFilter;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToBorder;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToBorder;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToBorder;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
        samplerInfo.compareEnable = VK_TRUE;
        samplerInfo.compareOp = vk::CompareOp::eLessOrEqual;
        m_ShadowSampler = vk::raii::Sampler(*m_Device, samplerInfo);
    }

    void VulkanRenderer::CreateShadowPipeline()
    {
        auto vertShaderModule = CreateShaderModule(ReadFile("shaders/shadow.vert.spv"));
        auto fragShaderModule = CreateShaderModule(ReadFile("shaders/shadow.frag.spv"));
        std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages = {
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "vertMain"),
            vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "fragMain")
        };

        // Same vertex layout as the raster pass
        vk::VertexInputBindingDescription binding(0, sizeof(Geometry::Vertex), vk::VertexInputRate::eVertex);
        std::array<vk::VertexInputAttributeDescription, 3> attributes{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Geometry::Vertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Geometry::Vertex, normal)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Geometry::Vertex, texCoord)),
        };
        vk::PipelineVertexInputStateCreateInfo vertexInput({}, binding, attributes);
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

        // Both faces cast shadows: thin single-sided surfaces (the awning)
        // would otherwise let light through from behind. The slope-scaled
        // bias keeps surfaces at a grazing angle to the sun from shadowing
        // themselves; the normal offset in lighting.slang does the rest.
        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.cullMode = vk::CullModeFlagBits::eNone;
        rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
        rasterizer.depthBiasEnable = VK_TRUE;
        rasterizer.depthBiasConstantFactor = 1.0f;
        rasterizer.depthBiasSlopeFactor = 1.5f;
        rasterizer.lineWidth = 1.0f;
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1);
        vk::PipelineDepthStencilStateCreateInfo depthStencil({}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess);
        vk::PipelineColorBlendStateCreateInfo colorBlending{};

        std::array<vk::DynamicState, 2> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates);

        // Depth only
        vk::PipelineRenderingCreateInfo renderingInfo({}, 0, nullptr, m_DepthFormat);

        vk::GraphicsPipelineCreateInfo pipelineInfo(
            {}, shaderStages, &vertexInput, &inputAssembly, nullptr, &viewportState, &rasterizer,
            &multisampling, &depthStencil, &colorBlending, &dynamicState, **m_PipelineLayout);
        pipelineInfo.pNext = &renderingInfo;
        m_ShadowPipeline = vk::raii::Pipeline(*m_Device, nullptr, pipelineInfo);
    }

    void VulkanRenderer::DrawShadowCasters(const VulkanModelData& modelData, const glm::mat4& modelMatrix, const glm::mat4& viewProjection,
                                           vk::ImageView target, uint32_t size)
    {
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        vk::RenderingAttachmentInfo depthAttachment{};
        depthAttachment.imageView = target;
        depthAttachment.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachment.clearValue = vk::ClearValue().setDepthStencil({ 1.0f, 0 });

        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea = vk::Rect2D({ 0, 0 }, { size, size });
        renderingInfo.layerCount = 1;
        renderingInfo.pDepthAttachment = &depthAttachment;

        commandBuffer.beginRendering(renderingInfo);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, **m_ShadowPipeline);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(size), static_cast<float>(size), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D({ 0, 0 }, { size, size }));
        for (const auto& mesh : modelData.meshes) {
            // The material set, for alpha-masked cut-outs
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, **m_PipelineLayout, 0, { mesh.materialData.descriptorSet }, nullptr);
            commandBuffer.bindVertexBuffers(0, *mesh.gpuMesh.vertexBuffer.buffer, { 0 });
            commandBuffer.bindIndexBuffer(*mesh.gpuMesh.indexBuffer.buffer, 0, vk::IndexType::eUint32);
            PushConstantData pushConstantData{
                .projectionMatrix = viewProjection,
                .viewMatrix = glm::mat4(1.0f),
                .modelMatrix = modelMatrix * mesh.localTransform,
            };
            commandBuffer.pushConstants<PushConstantData>(**m_PipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pushConstantData);
            commandBuffer.drawIndexed(mesh.gpuMesh.indexBuffer.indexCount, 1, 0, 0, 0);
        }
        commandBuffer.endRendering();
    }

    void VulkanRenderer::RecordShadowPass(const VulkanModelData& modelData, const glm::mat4& modelMatrix)
    {
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        const vk::ImageSubresourceRange depthRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

        // Shared by all frames in flight: wait for the previous frame's reads.
        // The pass clears it, so the old contents can be discarded.
        vk::ImageMemoryBarrier2 toAttachment{};
        toAttachment.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        toAttachment.srcAccessMask = {};
        toAttachment.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        toAttachment.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        toAttachment.oldLayout = vk::ImageLayout::eUndefined;
        toAttachment.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        toAttachment.image = *m_ShadowMap->image;
        toAttachment.subresourceRange = depthRange;
        vk::DependencyInfo beforeDependency{};
        beforeDependency.setImageMemoryBarriers(toAttachment);
        commandBuffer.pipelineBarrier2(beforeDependency);

        DrawShadowCasters(modelData, modelMatrix, SunViewProjection(), *m_ShadowMap->imageView, ShadowMapSize);

        // Depth writes -> the raster pass's reads
        vk::ImageMemoryBarrier2 toReadable{};
        toReadable.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests;
        toReadable.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        toReadable.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
        toReadable.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
        toReadable.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        toReadable.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        toReadable.image = *m_ShadowMap->image;
        toReadable.subresourceRange = depthRange;
        vk::DependencyInfo afterDependency{};
        afterDependency.setImageMemoryBarriers(toReadable);
        commandBuffer.pipelineBarrier2(afterDependency);
    }

    uint32_t VulkanRenderer::PointShadowCount() const
    {
        return m_Lighting.pointShadows ? static_cast<uint32_t>(std::min<size_t>(m_PointLights.size(), MaxShadowedPointLights)) : 0;
    }

    namespace
    {
        // World -> view space of each cube face, looking out from position, in
        // layer order +X, -X, +Y, -Y, +Z, -Z. Cube sampling (Vulkan spec,
        // "Cube Map Face Selection") reads a direction r on face +X at
        // s = -r.z / r.x, t = -r.y / r.x, and so on per face; rows 0 and 1 make
        // the view x and y exactly those numerators, row 2 points the view's -z
        // along the face axis, so after a 90 degree projection each pixel lands
        // where sampling will look for it. Row 0 of a framebuffer is NDC y = -1,
        // which is t = 0, so no flip is needed.
        glm::mat4 CubeFaceView(uint32_t face, const glm::vec3& position)
        {
            static const std::array<std::array<glm::vec3, 3>, 6> rows{ {
                { glm::vec3(0, 0, -1), glm::vec3(0, -1, 0), glm::vec3(-1, 0, 0) }, // +X
                { glm::vec3(0, 0, 1),  glm::vec3(0, -1, 0), glm::vec3(1, 0, 0) },  // -X
                { glm::vec3(1, 0, 0),  glm::vec3(0, 0, 1),  glm::vec3(0, -1, 0) }, // +Y
                { glm::vec3(1, 0, 0),  glm::vec3(0, 0, -1), glm::vec3(0, 1, 0) },  // -Y
                { glm::vec3(1, 0, 0),  glm::vec3(0, -1, 0), glm::vec3(0, 0, -1) }, // +Z
                { glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1) },  // -Z
            } };
            glm::mat4 view(1.0f);
            for (int row = 0; row < 3; row++) {
                for (int column = 0; column < 3; column++) {
                    view[column][row] = rows[face][row][column];
                }
                view[3][row] = -glm::dot(rows[face][row], position);
            }
            return view;
        }
    }

    void VulkanRenderer::CreatePointShadowMaps()
    {
        const vk::ImageSubresourceRange allFaces(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 6);
        auto commandBuffer = BeginSingleUseCommandBuffer();
        for (uint32_t light = 0; light < MaxShadowedPointLights; light++) {
            auto image = vk::raii::Image(*m_Device, vk::ImageCreateInfo(
                vk::ImageCreateFlagBits::eCubeCompatible,
                vk::ImageType::e2D,
                m_DepthFormat,
                vk::Extent3D{ PointShadowMapSize, PointShadowMapSize, 1 },
                1,
                6,
                vk::SampleCountFlagBits::e1,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
                vk::SharingMode::eExclusive,
                0,
                nullptr,
                vk::ImageLayout::eUndefined
            ));
            vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
            uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
            auto imageMemory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex));
            image.bindMemory(*imageMemory, 0);
            auto cubeView = vk::raii::ImageView(*m_Device, vk::ImageViewCreateInfo({}, *image, vk::ImageViewType::eCube, m_DepthFormat, {}, allFaces));

            PointShadowMap map{ AllocatedImage{ std::move(imageMemory), std::move(image), std::move(cubeView), PointShadowMapSize, PointShadowMapSize }, {} };
            for (uint32_t face = 0; face < 6; face++) {
                map.faces.emplace_back(*m_Device, vk::ImageViewCreateInfo({}, *map.cube.image, vk::ImageViewType::e2D, m_DepthFormat, {},
                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, face, 1)));
            }

            // Readable from the start, like the sun's map: lights without a
            // shadow still sit in the descriptor array. Cleared to the far
            // plane, so they would read as unshadowed.
            vk::ImageMemoryBarrier2 toClear{};
            toClear.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
            toClear.dstStageMask = vk::PipelineStageFlagBits2::eClear;
            toClear.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
            toClear.oldLayout = vk::ImageLayout::eUndefined;
            toClear.newLayout = vk::ImageLayout::eTransferDstOptimal;
            toClear.image = *map.cube.image;
            toClear.subresourceRange = allFaces;
            vk::DependencyInfo clearDependency{};
            clearDependency.setImageMemoryBarriers(toClear);
            commandBuffer.pipelineBarrier2(clearDependency);
            commandBuffer.clearDepthStencilImage(*map.cube.image, vk::ImageLayout::eTransferDstOptimal, vk::ClearDepthStencilValue(1.0f, 0), allFaces);

            vk::ImageMemoryBarrier2 toReadable{};
            toReadable.srcStageMask = vk::PipelineStageFlagBits2::eClear;
            toReadable.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
            toReadable.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            toReadable.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
            toReadable.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            toReadable.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            toReadable.image = *map.cube.image;
            toReadable.subresourceRange = allFaces;
            vk::DependencyInfo readableDependency{};
            readableDependency.setImageMemoryBarriers(toReadable);
            commandBuffer.pipelineBarrier2(readableDependency);

            m_PointShadowMaps.push_back(std::move(map));
        }
        EndSingleUseCommandBuffer(commandBuffer);
    }

    void VulkanRenderer::RecordPointShadowPasses(const VulkanModelData& modelData, const glm::mat4& modelMatrix)
    {
        // The scene doesn't move, so the maps only change with the lights
        std::vector<glm::vec3> positions;
        for (uint32_t i = 0; i < PointShadowCount(); i++) {
            positions.push_back(glm::vec3(m_PointLights[i].position));
        }
        if (positions == m_PointShadowPositions) {
            return;
        }
        m_PointShadowPositions = positions;

        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        const vk::ImageSubresourceRange allFaces(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 6);
        glm::mat4 projection = glm::perspective(glm::radians(90.0f), 1.0f, PointShadowNear, m_PointShadowFar);
        for (uint32_t light = 0; light < positions.size(); light++) {
            const PointShadowMap& map = m_PointShadowMaps[light];
            // Earlier frames may still be reading it; every face is cleared, so its contents can go
            vk::ImageMemoryBarrier2 toAttachment{};
            toAttachment.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            toAttachment.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
            toAttachment.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentRead | vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            toAttachment.oldLayout = vk::ImageLayout::eUndefined;
            toAttachment.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
            toAttachment.image = *map.cube.image;
            toAttachment.subresourceRange = allFaces;
            vk::DependencyInfo before{};
            before.setImageMemoryBarriers(toAttachment);
            commandBuffer.pipelineBarrier2(before);

            for (uint32_t face = 0; face < 6; face++) {
                DrawShadowCasters(modelData, modelMatrix, projection * CubeFaceView(face, positions[light]), *map.faces[face], PointShadowMapSize);
            }

            vk::ImageMemoryBarrier2 toReadable{};
            toReadable.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests;
            toReadable.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
            toReadable.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
            toReadable.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead;
            toReadable.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal;
            toReadable.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            toReadable.image = *map.cube.image;
            toReadable.subresourceRange = allFaces;
            vk::DependencyInfo after{};
            after.setImageMemoryBarriers(toReadable);
            commandBuffer.pipelineBarrier2(after);
        }
    }
} // namespace Renderer
} // namespace Momo
