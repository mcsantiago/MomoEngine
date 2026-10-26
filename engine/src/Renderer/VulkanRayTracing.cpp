// Compute shader ray tracing pass of VulkanRenderer: scene upload, pipeline,
// output image and per-frame recording. Kept apart from VulkanRenderer.cpp,
// which owns the device, swapchain and frame loop this builds on.
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Logging/Logger.h"
#include <glm/gtc/matrix_inverse.hpp>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <glm/gtc/packing.hpp>
#include <stb_image_write.h>

namespace Momo {
namespace Renderer {

    void VulkanRenderer::CheckRayTracingSupport()
    {
        m_RayTracingSupported = false;

        // Every pixel can sample a different texture of the scene's texture array
        if (!m_SupportsNonUniformTextureIndexing) {
            LOG_WARN("VulkanRenderer", "Device lacks shaderSampledImageArrayNonUniformIndexing. Ray tracing disabled.");
            return;
        }
        // Combined image samplers count against both the sampler and sampled image limits
        const vk::PhysicalDeviceLimits limits = m_PhysicalDevice->getProperties().limits;
        uint32_t textureLimit = std::min({ limits.maxPerStageDescriptorSamplers, limits.maxPerStageDescriptorSampledImages,
                                           limits.maxDescriptorSetSamplers, limits.maxDescriptorSetSampledImages });
        if (textureLimit < RayTracing::MaxRayTracingTextures) {
            LOG_WARN("VulkanRenderer", "Device allows {} textures per shader stage, ray tracing needs {}. Ray tracing disabled.",
                     textureLimit, RayTracing::MaxRayTracingTextures);
            return;
        }

        // The pass writes a storage image, then blits it into the swapchain image
        auto surfaceCapabilities = m_PhysicalDevice->getSurfaceCapabilitiesKHR(*m_Surface);
        if (!(surfaceCapabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)) {
            LOG_WARN("VulkanRenderer", "Swapchain images can't be transfer destinations. Ray tracing disabled.");
            return;
        }

        auto outputFeatures = m_PhysicalDevice->getFormatProperties(RayTracingOutputFormat).optimalTilingFeatures;
        if (!(outputFeatures & vk::FormatFeatureFlagBits::eStorageImage) || !(outputFeatures & vk::FormatFeatureFlagBits::eBlitSrc)) {
            LOG_WARN("VulkanRenderer", "{} can't be a storage image and blit source. Ray tracing disabled.", vk::to_string(RayTracingOutputFormat));
            return;
        }

        auto swapchainFeatures = m_PhysicalDevice->getFormatProperties(m_SwapchainImageFormat.format).optimalTilingFeatures;
        if (!(swapchainFeatures & vk::FormatFeatureFlagBits::eBlitDst)) {
            LOG_WARN("VulkanRenderer", "Swapchain format {} can't be a blit destination. Ray tracing disabled.", vk::to_string(m_SwapchainImageFormat.format));
            return;
        }

        auto accumulationFeatures = m_PhysicalDevice->getFormatProperties(RayTracingAccumulationFormat).optimalTilingFeatures;
        if (!(accumulationFeatures & vk::FormatFeatureFlagBits::eStorageImage)) {
            LOG_WARN("VulkanRenderer", "{} can't be a storage image. Ray tracing disabled.", vk::to_string(RayTracingAccumulationFormat));
            return;
        }

        m_RayTracingSupported = true;
        LOG_INFO("VulkanRenderer", "Compute ray tracing supported");
    }

    void VulkanRenderer::CreateRayTracingPipeline()
    {
        // Bindings 0-3: BVH nodes, triangle positions, triangle shading, materials.
        // 4: output image. 5: the scene's base color textures, as an array.
        // 6: accumulation image (running average across frames).
        // 7: traversal counters (only read by the counting shader variant).
        // 8-10: captured rays, their counts per queue, and trace-only hits
        // (only used by the trace-only experiment). 11: point lights.
        std::array<vk::DescriptorSetLayoutBinding, 12> bindings{};
        for (uint32_t i = 0; i < bindings.size(); i++) {
            bindings[i].binding = i;
            bindings[i].descriptorCount = 1;
            bindings[i].descriptorType = vk::DescriptorType::eStorageBuffer;
            bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
        }
        bindings[4].descriptorType = vk::DescriptorType::eStorageImage;
        bindings[5].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        bindings[5].descriptorCount = RayTracing::MaxRayTracingTextures;
        bindings[6].descriptorType = vk::DescriptorType::eStorageImage;
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        m_RayTracingSetLayout = vk::raii::DescriptorSetLayout(*m_Device, layoutInfo);

        // The pass has its own pool, sized for its single set
        m_RayTracingDescriptorAllocator.Init(*m_Device, DescriptorPoolSizes{
            .maxSets = 1,
            .sizes = {
                { vk::DescriptorType::eStorageBuffer, 9 },
                { vk::DescriptorType::eStorageImage, 2 },
                { vk::DescriptorType::eCombinedImageSampler, RayTracing::MaxRayTracingTextures },
            },
        });

        vk::PushConstantRange pushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, sizeof(RayTracingPushConstants));
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, **m_RayTracingSetLayout, pushConstantRange);
        m_RayTracingPipelineLayout = vk::raii::PipelineLayout(*m_Device, pipelineLayoutInfo);

        auto createPipeline = [this](const std::string& shaderPath, vk::PipelineLayout layout) {
            auto shaderCode = ReadFile(shaderPath);
            vk::raii::ShaderModule shaderModule = CreateShaderModule(shaderCode);
            vk::ComputePipelineCreateInfo pipelineInfo(
                m_PipelineStatisticsEnabled ? vk::PipelineCreateFlagBits::eCaptureStatisticsKHR : vk::PipelineCreateFlags{},
                vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *shaderModule, "compMain"),
                layout
            );
            return vk::raii::Pipeline(*m_Device, nullptr, pipelineInfo);
        };
        m_RayTracingPipeline = createPipeline("shaders/raytrace.comp.spv", **m_RayTracingPipelineLayout);
        // Same shader compiled with COUNT_TRAVERSAL / CAPTURE_RAYS, for benchmarks
        m_RayTracingCountingPipeline = createPipeline("shaders/raytrace_counters.comp.spv", **m_RayTracingPipelineLayout);
        m_RayTracingCapturePipeline = createPipeline("shaders/raytrace_capture.comp.spv", **m_RayTracingPipelineLayout);

        // The trace-only kernel shares the descriptor set but has its own push constants
        vk::PushConstantRange traceOnlyRange(vk::ShaderStageFlagBits::eCompute, 0, 4 * sizeof(uint32_t));
        vk::PipelineLayoutCreateInfo traceOnlyLayoutInfo({}, **m_RayTracingSetLayout, traceOnlyRange);
        m_TraceOnlyPipelineLayout = vk::raii::PipelineLayout(*m_Device, traceOnlyLayoutInfo);
        m_TraceOnlyPipeline = createPipeline("shaders/traceonly.comp.spv", **m_TraceOnlyPipelineLayout);
        LogPipelineStatistics(*m_RayTracingPipeline, "raytrace");
        LogPipelineStatistics(*m_TraceOnlyPipeline, "traceonly");

        std::array<uint32_t, 2 * TraversalCounterCount> zeros{};
        m_TraversalCounterBuffer.emplace(CreateBuffer(zeros.data(), sizeof(zeros), vk::BufferUsageFlagBits::eStorageBuffer, 0));
        LOG_INFO("VulkanRenderer", "Ray tracing pipeline created successfully");
    }

    AllocatedImage VulkanRenderer::CreateRayTracingStorageImage(vk::Format format, vk::ImageUsageFlags usage)
    {
        auto image = vk::raii::Image(*m_Device, vk::ImageCreateInfo(
            {},
            vk::ImageType::e2D,
            format,
            vk::Extent3D{ m_SwapchainExtent.width, m_SwapchainExtent.height, 1 },
            1,
            1,
            vk::SampleCountFlagBits::e1,
            vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eStorage | usage,
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
            format,
            {},
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
        );
        auto imageView = vk::raii::ImageView(*m_Device, viewInfo);

        return AllocatedImage{ std::move(imageMemory), std::move(image), std::move(imageView), m_SwapchainExtent.width, m_SwapchainExtent.height };
    }

    void VulkanRenderer::CreateRayTracingImages()
    {
        m_RayTracingOutput.emplace(CreateRayTracingStorageImage(RayTracingOutputFormat, vk::ImageUsageFlagBits::eTransferSrc));
        m_RayTracingAccumulation.emplace(CreateRayTracingStorageImage(RayTracingAccumulationFormat, {}));
        // Capture queues hold one ray per pixel; reallocated for the new size when next needed
        m_CapturedRays.reset();
        m_CaptureCounts.reset();
        m_TraceHits.reset();
        m_CaptureRays = false;
        // The old running average was for the old size
        m_AccumulatedSamples = 0;
    }

    AllocatedBuffer VulkanRenderer::CreateDeviceLocalBuffer(const void* data, vk::DeviceSize size, vk::BufferUsageFlags usage)
    {
        AllocatedBuffer stagingBuffer = CreateBuffer(data, size, vk::BufferUsageFlagBits::eTransferSrc, 0);

        auto buffer = vk::raii::Buffer(*m_Device, vk::BufferCreateInfo(
            {}, size, usage | vk::BufferUsageFlagBits::eTransferDst,
            vk::SharingMode::eExclusive
        ));
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto bufferMemory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex));
        buffer.bindMemory(*bufferMemory, 0);

        auto commandBuffer = BeginSingleUseCommandBuffer();
        commandBuffer.copyBuffer(*stagingBuffer.buffer, *buffer, vk::BufferCopy(0, 0, size));

        // Waiting for the queue to go idle doesn't make the copy visible to
        // shaders; this barrier does, for every later command in submission order
        vk::MemoryBarrier2 afterCopy(
            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageRead
        );
        vk::DependencyInfo dependencyInfo{};
        dependencyInfo.setMemoryBarriers(afterCopy);
        commandBuffer.pipelineBarrier2(dependencyInfo);
        EndSingleUseCommandBuffer(commandBuffer);

        return AllocatedBuffer{ std::move(bufferMemory), std::move(buffer), 0 };
    }

    void VulkanRenderer::UploadRayTracingScene(RayTracing::SceneGeometry geometry, std::vector<vk::ImageView> textureViews)
    {
        if (!m_RayTracingSupported) {
            LOG_WARN("VulkanRenderer", "Ray tracing is not supported on this device. Skipping scene upload.");
            return;
        }
        if (geometry.positions.empty()) {
            LOG_WARN("VulkanRenderer", "Ray tracing scene has no triangles. Skipping scene upload.");
            return;
        }
        if (textureViews.size() != geometry.textures.size()) {
            LOG_ERROR("VulkanRenderer", "Got {} texture views for {} ray tracing textures. Skipping scene upload.",
                      textureViews.size(), geometry.textures.size());
            return;
        }

        // Frames in flight may still be reading the previous scene's buffers
        m_Device->waitIdle();
        m_RayTracingScene.reset();

        auto upload = [this](const auto& items) {
            return CreateDeviceLocalBuffer(items.data(), sizeof(items[0]) * items.size(), vk::BufferUsageFlagBits::eStorageBuffer);
        };
        size_t totalBytes = sizeof(RayTracing::CompressedWideNode) * geometry.bvh.nodes.size()
                          + sizeof(RayTracing::TrianglePositions) * geometry.positions.size()
                          + sizeof(RayTracing::TriangleShading) * geometry.shading.size()
                          + sizeof(RayTracing::RayTracingMaterial) * geometry.materials.size()
                          + sizeof(RayTracing::RayTracingPointLight) * geometry.pointLights.size();
        // A storage buffer can't be empty; the shader reads only pointLightCount entries
        std::vector<RayTracing::RayTracingPointLight> pointLights = geometry.pointLights;
        if (pointLights.empty()) {
            pointLights.push_back(RayTracing::RayTracingPointLight{});
        }
        m_RayTracingScene.emplace(RayTracingSceneBuffers{
            .nodes = upload(geometry.bvh.nodes),
            .positions = upload(geometry.positions),
            .shading = upload(geometry.shading),
            .materials = upload(geometry.materials),
            .pointLights = upload(pointLights),
            .pointLightCount = static_cast<uint32_t>(geometry.pointLights.size()),
            .stats = geometry.bvh.stats,
            .triangleCount = geometry.positions.size(),
            .gpuBytes = totalBytes,
        });
        m_RayTracingTextureViews = std::move(textureViews);
        m_AccumulatedSamples = 0;

        if (!m_RayTracingDescriptorSet) {
            m_RayTracingDescriptorSet = m_RayTracingDescriptorAllocator.AllocateDescriptorSet(**m_RayTracingSetLayout);
        }
        WriteRayTracingDescriptorSet();

        LOG_INFO("VulkanRenderer", "Uploaded ray tracing scene: {} triangles, {} materials, {} textures, {} point lights, {:.1f} MB",
                 geometry.positions.size(), geometry.materials.size(), geometry.textures.size(), geometry.pointLights.size(),
                 totalBytes / (1024.0 * 1024.0));
    }

    void VulkanRenderer::WriteRayTracingDescriptorSet()
    {
        if (!m_RayTracingScene || !m_RayTracingOutput || !m_RayTracingAccumulation) {
            return;
        }

        std::array<vk::DescriptorBufferInfo, 4> bufferInfos{
            vk::DescriptorBufferInfo(*m_RayTracingScene->nodes.buffer, 0, vk::WholeSize),
            vk::DescriptorBufferInfo(*m_RayTracingScene->positions.buffer, 0, vk::WholeSize),
            vk::DescriptorBufferInfo(*m_RayTracingScene->shading.buffer, 0, vk::WholeSize),
            vk::DescriptorBufferInfo(*m_RayTracingScene->materials.buffer, 0, vk::WholeSize),
        };
        vk::DescriptorImageInfo imageInfo({}, *m_RayTracingOutput->imageView, vk::ImageLayout::eGeneral);
        vk::DescriptorImageInfo accumulationInfo({}, *m_RayTracingAccumulation->imageView, vk::ImageLayout::eGeneral);

        // Every slot must hold a valid texture; unused ones repeat slot 0 (white)
        std::vector<vk::DescriptorImageInfo> textureInfos(RayTracing::MaxRayTracingTextures,
            vk::DescriptorImageInfo(*m_TextureSampler, m_RayTracingTextureViews[0], vk::ImageLayout::eShaderReadOnlyOptimal));
        for (size_t i = 0; i < m_RayTracingTextureViews.size(); i++) {
            textureInfos[i].imageView = m_RayTracingTextureViews[i];
        }

        vk::DescriptorBufferInfo counterInfo(*m_TraversalCounterBuffer->buffer, 0, vk::WholeSize);
        vk::DescriptorBufferInfo pointLightInfo(*m_RayTracingScene->pointLights.buffer, 0, vk::WholeSize);

        std::array<vk::WriteDescriptorSet, 9> writes{};
        for (uint32_t i = 0; i < bufferInfos.size(); i++) {
            writes[i] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, i, 0, vk::DescriptorType::eStorageBuffer, nullptr, bufferInfos[i]);
        }
        writes[4] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, 4, 0, vk::DescriptorType::eStorageImage, imageInfo);
        writes[5] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, 5, 0, vk::DescriptorType::eCombinedImageSampler, textureInfos);
        writes[6] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, 6, 0, vk::DescriptorType::eStorageImage, accumulationInfo);
        writes[7] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, 7, 0, vk::DescriptorType::eStorageBuffer, nullptr, counterInfo);
        writes[8] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, 11, 0, vk::DescriptorType::eStorageBuffer, nullptr, pointLightInfo);
        m_Device->updateDescriptorSets(writes, nullptr);
    }

    // What the driver's compiler made of a pipeline's shader: on NVIDIA this
    // includes registers per thread, which limit how many threads run at once
    void VulkanRenderer::LogPipelineStatistics(const vk::raii::Pipeline& pipeline, const std::string& name)
    {
        if (!m_PipelineStatisticsEnabled) {
            return;
        }
        auto executables = m_Device->getPipelineExecutablePropertiesKHR(vk::PipelineInfoKHR(*pipeline));
        for (uint32_t i = 0; i < executables.size(); i++) {
            LOG_INFO("VulkanRenderer", "Pipeline '{}' executable {} ({}), subgroup size {}:", name, i,
                     executables[i].name.data(), executables[i].subgroupSize);
            for (const auto& statistic : m_Device->getPipelineExecutableStatisticsKHR(vk::PipelineExecutableInfoKHR(*pipeline, i))) {
                std::string value;
                switch (statistic.format) {
                    case vk::PipelineExecutableStatisticFormatKHR::eBool32: value = statistic.value.b32 ? "true" : "false"; break;
                    case vk::PipelineExecutableStatisticFormatKHR::eInt64: value = std::to_string(statistic.value.i64); break;
                    case vk::PipelineExecutableStatisticFormatKHR::eUint64: value = std::to_string(statistic.value.u64); break;
                    case vk::PipelineExecutableStatisticFormatKHR::eFloat64: value = std::to_string(statistic.value.f64); break;
                }
                LOG_INFO("VulkanRenderer", "  {}: {} ({})", statistic.name.data(), value, statistic.description.data());
            }
        }
    }

    AllocatedBuffer VulkanRenderer::CreateUninitializedBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProperties)
    {
        auto buffer = vk::raii::Buffer(*m_Device, vk::BufferCreateInfo({}, size, usage, vk::SharingMode::eExclusive));
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits, memoryProperties);
        auto memory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex));
        buffer.bindMemory(*memory, 0);
        return AllocatedBuffer{ std::move(memory), std::move(buffer), 0 };
    }

    void VulkanRenderer::EnsureCaptureBuffers()
    {
        if (m_CapturedRays) {
            return;
        }
        m_CaptureCapacity = m_SwapchainExtent.width * m_SwapchainExtent.height;
        constexpr vk::DeviceSize capturedRaySize = 2 * 4 * sizeof(float); // CapturedRay: two float4s
        m_CapturedRays.emplace(CreateUninitializedBuffer(CaptureQueueCount * m_CaptureCapacity * capturedRaySize,
            vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal));
        m_CaptureCounts.emplace(CreateUninitializedBuffer(CaptureQueueCount * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent));
        m_TraceHits.emplace(CreateUninitializedBuffer(vk::DeviceSize(m_CaptureCapacity) * 2 * sizeof(float),
            vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal));

        std::array<vk::DescriptorBufferInfo, 3> infos{
            vk::DescriptorBufferInfo(*m_CapturedRays->buffer, 0, vk::WholeSize),
            vk::DescriptorBufferInfo(*m_CaptureCounts->buffer, 0, vk::WholeSize),
            vk::DescriptorBufferInfo(*m_TraceHits->buffer, 0, vk::WholeSize),
        };
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (uint32_t i = 0; i < infos.size(); i++) {
            writes[i] = vk::WriteDescriptorSet(m_RayTracingDescriptorSet, 8 + i, 0, vk::DescriptorType::eStorageBuffer, nullptr, infos[i]);
        }
        m_Device->updateDescriptorSets(writes, nullptr);
        LOG_INFO("VulkanRenderer", "Allocated ray capture buffers: {} queues of {} rays", CaptureQueueCount, m_CaptureCapacity);
    }

    void VulkanRenderer::SetRayCapture(bool enabled)
    {
        if (enabled && !m_CaptureRays) {
            // Frames in flight may still use the descriptor set or the buffers
            m_Device->waitIdle();
            EnsureCaptureBuffers();
            void* counts = m_CaptureCounts->memory.mapMemory(0, CaptureQueueCount * sizeof(uint32_t));
            std::memset(counts, 0, CaptureQueueCount * sizeof(uint32_t));
            m_CaptureCounts->memory.unmapMemory();
        }
        m_CaptureRays = enabled;
    }

    std::vector<TraceOnlyQueueResult> VulkanRenderer::RunTraceOnlyExperiment(uint32_t repetitions)
    {
        std::vector<TraceOnlyQueueResult> results;
        if (!m_CaptureCounts || !m_GpuTimingEnabled) {
            return results;
        }
        m_Device->waitIdle();

        std::array<uint32_t, CaptureQueueCount> counts;
        const auto* mapped = static_cast<const uint32_t*>(m_CaptureCounts->memory.mapMemory(0, sizeof(counts)));
        std::copy(mapped, mapped + CaptureQueueCount, counts.begin());
        m_CaptureCounts->memory.unmapMemory();

        vk::QueryPoolCreateInfo queryPoolInfo({}, vk::QueryType::eTimestamp, 2);
        vk::raii::QueryPool queryPool(*m_Device, queryPoolInfo);

        for (uint32_t queue = 0; queue < CaptureQueueCount; queue++) {
            uint32_t rays = std::min(counts[queue], m_CaptureCapacity);
            if (rays == 0) {
                continue;
            }
            bool shadow = queue >= CaptureBounceLimit;
            std::array<uint32_t, 4> pushConstants{ queue, shadow ? 1u : 0u, m_CaptureCapacity, 0 };

            // One extra, discarded run first: the GPU may have clocked down while idle
            std::vector<double> times;
            for (uint32_t run = 0; run <= repetitions; run++) {
                auto commandBuffer = BeginSingleUseCommandBuffer();
                commandBuffer.resetQueryPool(*queryPool, 0, 2);
                commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *queryPool, 0);
                commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **m_TraceOnlyPipeline);
                commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **m_TraceOnlyPipelineLayout, 0, m_RayTracingDescriptorSet, nullptr);
                commandBuffer.pushConstants<std::array<uint32_t, 4>>(**m_TraceOnlyPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);
                commandBuffer.dispatch((rays + 63) / 64, 1, 1);
                commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *queryPool, 1);
                EndSingleUseCommandBuffer(commandBuffer);

                auto [result, ticks] = queryPool.getResults<uint64_t>(0, 2, 2 * sizeof(uint64_t), sizeof(uint64_t),
                    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
                if (run > 0 && result == vk::Result::eSuccess) {
                    times.push_back(((ticks[1] - ticks[0]) & m_TimestampValidBitmask) * m_TimestampPeriodNs / 1'000'000.0);
                }
            }
            std::sort(times.begin(), times.end());
            results.push_back({ shadow, queue % CaptureBounceLimit, rays, times.empty() ? 0.0 : times[times.size() / 2] });
        }
        return results;
    }

    TraversalCounts VulkanRenderer::ReadAndResetTraversalCounters()
    {
        TraversalCounts counts{};
        if (!m_TraversalCounterBuffer) {
            return counts;
        }
        // Every frame that counted has to finish first
        m_Device->waitIdle();

        constexpr vk::DeviceSize size = 2 * TraversalCounterCount * sizeof(uint32_t);
        auto* words = static_cast<uint32_t*>(m_TraversalCounterBuffer->memory.mapMemory(0, size));
        for (uint32_t i = 0; i < TraversalCounterCount; i++) {
            counts.values[i] = (uint64_t(words[2 * i + 1]) << 32) | words[2 * i];
        }
        std::memset(words, 0, size);
        m_TraversalCounterBuffer->memory.unmapMemory();
        return counts;
    }

    bool VulkanRenderer::SaveRayTracingOutput(const std::filesystem::path& path)
    {
        if (!m_RayTracingOutput) {
            return false;
        }
        // The last frame's blit left the output image as a transfer source
        m_Device->waitIdle();

        const uint32_t width = m_RayTracingOutput->width;
        const uint32_t height = m_RayTracingOutput->height;
        const vk::DeviceSize size = vk::DeviceSize(width) * height * 4 * sizeof(uint16_t); // rgba16f

        auto buffer = vk::raii::Buffer(*m_Device, vk::BufferCreateInfo({}, size, vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive));
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        uint32_t memoryTypeIndex = FindMemoryType(memRequirements.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        auto memory = vk::raii::DeviceMemory(*m_Device, vk::MemoryAllocateInfo(memRequirements.size, memoryTypeIndex));
        buffer.bindMemory(*memory, 0);

        auto commandBuffer = BeginSingleUseCommandBuffer();
        vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1),
                                   vk::Offset3D{ 0, 0, 0 }, vk::Extent3D{ width, height, 1 });
        commandBuffer.copyImageToBuffer(*m_RayTracingOutput->image, vk::ImageLayout::eTransferSrcOptimal, *buffer, region);
        // Make the copy visible to the host read below
        vk::MemoryBarrier2 toHost(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
        vk::DependencyInfo dependencyInfo{};
        dependencyInfo.setMemoryBarriers(toHost);
        commandBuffer.pipelineBarrier2(dependencyInfo);
        EndSingleUseCommandBuffer(commandBuffer);

        // Half floats in linear light (already tonemapped) -> 8-bit sRGB, which
        // is what the blit to the swapchain does for display
        auto toSrgb8 = [](float linear) {
            linear = std::clamp(linear, 0.0f, 1.0f);
            float encoded = linear <= 0.0031308f ? linear * 12.92f : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            return static_cast<uint8_t>(std::lround(encoded * 255.0f));
        };
        const auto* halves = static_cast<const uint16_t*>(memory.mapMemory(0, size));
        std::vector<uint8_t> pixels(size_t(width) * height * 3);
        for (size_t i = 0; i < size_t(width) * height; i++) {
            for (size_t channel = 0; channel < 3; channel++) {
                pixels[i * 3 + channel] = toSrgb8(glm::unpackHalf1x16(halves[i * 4 + channel]));
            }
        }
        memory.unmapMemory();

        if (!stbi_write_jpg(path.string().c_str(), int(width), int(height), 3, pixels.data(), 90)) {
            LOG_ERROR("VulkanRenderer", "Couldn't write screenshot {}", path.string());
            return false;
        }
        LOG_INFO("VulkanRenderer", "Saved ray traced frame to {}", path.string());
        return true;
    }

    void VulkanRenderer::RecordRayTracingPass(uint32_t imageIndex, const glm::mat4& viewMatrix)
    {
        auto& commandBuffer = m_CommandBuffers[m_CurrentFrame];
        const vk::ImageSubresourceRange colorRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

        // Keep averaging while the camera holds still; any change (moving,
        // turning, resizing) makes the old samples wrong, so start over
        glm::mat4 viewProjection = m_ProjectionMatrix * viewMatrix;
        if (viewProjection != m_AccumulationViewProjection) {
            m_AccumulationViewProjection = viewProjection;
            m_AccumulatedSamples = 0;
        }
        bool restartAccumulation = m_AccumulatedSamples == 0;
        m_AccumulatedSamples++;

        // The output image is shared by all frames in flight, so wait for the
        // previous frame's blit to finish reading it. Every pixel is rewritten,
        // so its old contents can be discarded (Undefined).
        vk::ImageMemoryBarrier2 toGeneral{};
        toGeneral.srcStageMask = vk::PipelineStageFlagBits2::eBlit;
        toGeneral.srcAccessMask = {};
        toGeneral.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        toGeneral.dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
        toGeneral.oldLayout = vk::ImageLayout::eUndefined;
        toGeneral.newLayout = vk::ImageLayout::eGeneral;
        toGeneral.image = *m_RayTracingOutput->image;
        toGeneral.subresourceRange = colorRange;

        // The accumulation image is read and written by every frame, so each
        // dispatch waits for the previous one's writes. On a restart the
        // shader doesn't read it, so the old contents can be discarded.
        vk::ImageMemoryBarrier2 accumulationBarrier{};
        accumulationBarrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        accumulationBarrier.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
        accumulationBarrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        accumulationBarrier.dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite;
        accumulationBarrier.oldLayout = restartAccumulation ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral;
        accumulationBarrier.newLayout = vk::ImageLayout::eGeneral;
        accumulationBarrier.image = *m_RayTracingAccumulation->image;
        accumulationBarrier.subresourceRange = colorRange;

        std::array<vk::ImageMemoryBarrier2, 2> beforeDispatchBarriers{ toGeneral, accumulationBarrier };
        vk::DependencyInfo beforeDispatch{};
        beforeDispatch.setImageMemoryBarriers(beforeDispatchBarriers);
        commandBuffer.pipelineBarrier2(beforeDispatch);

        glm::vec3 cameraPosition = glm::inverse(viewMatrix)[3];
        RayTracingPushConstants pushConstants{
            .inverseViewProjection = glm::inverse(viewProjection),
            .cameraPosition = glm::vec4(cameraPosition, 1.0f),
            .frameIndex = m_RayTracingFrameIndex++,
            .sampleCount = m_AccumulatedSamples,
            .maxBounces = m_RayTracingMaxBounces,
            .pointLightCount = m_RayTracingScene ? m_RayTracingScene->pointLightCount : 0,
        };
        const vk::raii::Pipeline& pipeline = m_CaptureRays ? *m_RayTracingCapturePipeline
                                           : m_CountTraversal ? *m_RayTracingCountingPipeline : *m_RayTracingPipeline;
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, **m_RayTracingPipelineLayout, 0, m_RayTracingDescriptorSet, nullptr);
        commandBuffer.pushConstants<RayTracingPushConstants>(**m_RayTracingPipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, pushConstants);
        // Time the dispatch alone, for benchmarks. Each timestamp is written once
        // all earlier compute work in this command buffer has finished.
        uint32_t firstQuery = m_CurrentFrame * QUERIES_PER_FRAME;
        if (m_GpuTimingEnabled) {
            commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, **m_TimestampQueryPool, firstQuery + 2);
        }
        // 8x8 threads per group, matching [numthreads] in raytrace.slang
        commandBuffer.dispatch((m_SwapchainExtent.width + 7) / 8, (m_SwapchainExtent.height + 7) / 8, 1);
        if (m_GpuTimingEnabled) {
            commandBuffer.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, **m_TimestampQueryPool, firstQuery + 3);
        }
        if (m_CaptureRays) {
            // Make the captured rays and counts visible to the trace-only kernel and the host
            vk::MemoryBarrier2 captured(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
                                        vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eHost,
                                        vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eHostRead);
            vk::DependencyInfo captureDependency{};
            captureDependency.setMemoryBarriers(captured);
            commandBuffer.pipelineBarrier2(captureDependency);
        }
        if (m_CountTraversal) {
            // Make the counters' atomic adds visible to ReadAndResetTraversalCounters
            vk::MemoryBarrier2 toHost(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
                                      vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
            vk::DependencyInfo counterDependency{};
            counterDependency.setMemoryBarriers(toHost);
            commandBuffer.pipelineBarrier2(counterDependency);
        }

        // Output: shader writes -> blit reads
        vk::ImageMemoryBarrier2 toTransferSrc{};
        toTransferSrc.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
        toTransferSrc.srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite;
        toTransferSrc.dstStageMask = vk::PipelineStageFlagBits2::eBlit;
        toTransferSrc.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
        toTransferSrc.oldLayout = vk::ImageLayout::eGeneral;
        toTransferSrc.newLayout = vk::ImageLayout::eTransferSrcOptimal;
        toTransferSrc.image = *m_RayTracingOutput->image;
        toTransferSrc.subresourceRange = colorRange;

        // Swapchain image: becomes a blit target. The source stage chains with the
        // image-available semaphore, which the submit waits on at the transfer stage.
        vk::ImageMemoryBarrier2 toTransferDst{};
        toTransferDst.srcStageMask = vk::PipelineStageFlagBits2::eBlit;
        toTransferDst.srcAccessMask = {};
        toTransferDst.dstStageMask = vk::PipelineStageFlagBits2::eBlit;
        toTransferDst.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
        toTransferDst.oldLayout = vk::ImageLayout::eUndefined;
        toTransferDst.newLayout = vk::ImageLayout::eTransferDstOptimal;
        toTransferDst.image = m_SwapchainImages[imageIndex];
        toTransferDst.subresourceRange = colorRange;

        std::array<vk::ImageMemoryBarrier2, 2> beforeBlitBarriers{ toTransferSrc, toTransferDst };
        vk::DependencyInfo beforeBlit{};
        beforeBlit.setImageMemoryBarriers(beforeBlitBarriers);
        commandBuffer.pipelineBarrier2(beforeBlit);

        // Same size, so this is a copy plus the float -> sRGB conversion of the swapchain format
        vk::ImageSubresourceLayers colorLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
        std::array<vk::Offset3D, 2> extent{ vk::Offset3D{ 0, 0, 0 },
            vk::Offset3D{ static_cast<int32_t>(m_SwapchainExtent.width), static_cast<int32_t>(m_SwapchainExtent.height), 1 } };
        vk::ImageBlit blitRegion(colorLayers, extent, colorLayers, extent);
        commandBuffer.blitImage(
            *m_RayTracingOutput->image, vk::ImageLayout::eTransferSrcOptimal,
            m_SwapchainImages[imageIndex], vk::ImageLayout::eTransferDstOptimal,
            blitRegion, vk::Filter::eNearest
        );
    }

} // namespace Renderer
} // namespace Momo
