#pragma once
#include <vulkan/vulkan_raii.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <array>
#include <optional>
#include <cstdint>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Vulkan depth [0, 1] range
#include <glm/glm.hpp>
#include "Momo/Window.h"
#include "Momo/Geometry/Vertex.h"
#include "VulkanMeshData.h"
#include "Momo/Assets/ModelData.h"
#include "Momo/Renderer/DescriptorAllocator.h"
#include "Momo/Profiling/Timer.h"
#include "Momo/RayTracing/SceneGeometry.h"
#include "Momo/Renderer/Lighting.h"

namespace Momo {
namespace Renderer {
struct DrawFrameTime
{
    float recordMs, submitMs, fenceWaitMs, gpuFrameMs;
    // GPU time of the ray tracing dispatch alone. GPU times arrive
    // MAX_FRAMES_IN_FLIGHT frames late (read back when the frame slot comes
    // round again), so this describes an earlier frame; false if that frame
    // wasn't ray traced.
    bool  hasRayTracingMs;
    float rayTracingMs;
};

enum class RenderMode
{
    Raster,    // Rasterized, Lambert shading; no shadows or bounce light
    Toon,      // Rasterized, toon shading with outlines
    RayTraced  // Path traced
};

inline const char* RenderModeName(RenderMode mode)
{
    switch (mode)
    {
    case RenderMode::Raster: return "raster";
    case RenderMode::Toon: return "toon";
    case RenderMode::RayTraced: return "ray traced";
    }
    return "unknown";
}

// Toon shading parameters as lighting.slang reads them (std140)
struct ToonUniforms
{
    glm::vec4 shadowTint;
    glm::vec4 rimColor;
    glm::vec4 outlineColor;
    uint32_t  bands;
    float     bandSoftness;
    float     specularStrength;
    float     specularSize;
    float     rimStrength;
    float     rimWidth;
    float     outlineWidth;
    float     depthThreshold;
    float     normalThreshold;
    float     padding[3];
};
static_assert(sizeof(ToonUniforms) == 96, "ToonUniforms must match ToonParams in lighting.slang");

// Per-frame uniforms of every pipeline (lighting.slang, set 1 binding 0; std140)
struct FrameUniforms
{
    glm::mat4 inverseViewProjection;
    glm::mat4 sunViewProjection;
    glm::vec4 cameraPosition;
    glm::vec4 sunDirection;
    glm::vec4 sunIrradiance;
    glm::vec4 skyGround;
    glm::vec4 skyHorizon;
    glm::vec4 skyZenith;
    float     exposure;
    uint32_t  pointLightCount;
    uint32_t  shadingModel; // ShadingModel
    float     nearPlane;
    float     farPlane;
    uint32_t  shadowsEnabled;
    float     shadowNormalOffset; // World units
    float     shadowFilterRadius; // Shadow map texels
    ToonUniforms toon;
};
static_assert(sizeof(FrameUniforms) == 352, "FrameUniforms must match lighting.slang");

// A point light as lighting.slang reads it (set 1 binding 1; std430)
struct GpuPointLight
{
    glm::vec4 position;  // w unused
    glm::vec4 intensity; // rgb: radiant intensity (see Assets::PointLight); a unused
};
static_assert(sizeof(GpuPointLight) == 32, "GpuPointLight must match lighting.slang");

// Push constants of raytrace.slang (std430-compatible; at most the guaranteed 128 bytes)
struct RayTracingPushConstants
{
    glm::mat4 inverseViewProjection; // NDC -> world, to build each pixel's camera ray
    glm::vec4 cameraPosition;        // w unused
    uint32_t  frameIndex;            // Changes every ray traced frame; seeds the random numbers
    uint32_t  sampleCount;           // Samples in the running average including this frame; 1 = start over
    uint32_t  maxBounces;            // Bounces after the camera ray's first hit
    uint32_t  padding;
};
static_assert(sizeof(RayTracingPushConstants) == 96, "RayTracingPushConstants must match raytrace.slang");

// Totals from the ray tracing shader's counting variant (raytrace.slang, COUNT_TRAVERSAL)
enum TraversalCounter : uint32_t
{
    ClosestRays,   // Camera and bounce rays (closest hit)
    ShadowRays,    // Any-hit rays toward the sun
    Steps,         // Traversal loop iterations
    NodeReads,     // BVH nodes read
    BoxTests,
    TriangleTests,
    AlphaTests,
    TraversalCounterCount
};

struct TraversalCounts
{
    uint64_t values[TraversalCounterCount];
};

// Trace-only experiment: one capture queue (rays of one kind and bounce)
// traced again by the small trace-only kernel
struct TraceOnlyQueueResult
{
    bool shadow;     // Shadow (any-hit) rays; otherwise camera/bounce (closest-hit) rays
    uint32_t bounce;
    uint32_t rays;
    double medianMs; // GPU time of tracing the queue once
};

// GPU copies of RayTracing::SceneGeometry, in device-local storage buffers
struct RayTracingSceneBuffers
{
    AllocatedBuffer nodes;
    AllocatedBuffer positions;
    AllocatedBuffer shading;
    AllocatedBuffer materials;
    RayTracing::WideBvhStats stats;
    size_t triangleCount;
    size_t gpuBytes; // All four buffers
};

class VulkanRenderer
{
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    // Timestamp queries per frame slot: frame start/end, ray tracing dispatch start/end
    static constexpr uint32_t QUERIES_PER_FRAME = 4;

    void Init(const IWindow& window);
    void Shutdown();
    // Sets up the Dear ImGui Vulkan backend; the caller owns the ImGui context
    // and the platform backend. Each frame then draws ImGui::GetDrawData() on top.
    void InitImGui();
    DrawFrameTime RenderFrame(Renderer::VulkanModelData& modelData, glm::mat4 viewMatrix, glm::mat4 modelMatrix, RenderMode mode);

    // Uploads the scene's ray tracing geometry and keeps only the GPU copy.
    // textureViews[i] is the GPU image of geometry.textures[i]; the caller keeps
    // the images alive while the scene is in use. Replaces any previous scene.
    void UploadRayTracingScene(RayTracing::SceneGeometry geometry, std::vector<vk::ImageView> textureViews);
    // The scene's point lights, in world space, for every render path. Replaces any previous ones.
    void SetPointLights(const std::vector<Assets::PointLight>& lights);
    // Used from the next frame on
    void SetLighting(const SceneLighting& lighting) { m_Lighting = lighting; }
    void SetToonSettings(const ToonSettings& toon) { m_Toon = toon; }
    // World-space box around the scene; the sun's shadow map covers it
    void SetShadowBounds(const glm::vec3& min, const glm::vec3& max);
    // False if the device can't present a ray traced image or no scene was uploaded
    bool IsRayTracingAvailable() const { return m_RayTracingSupported && m_RayTracingScene.has_value(); }
    uint32_t GetRayTracingMaxBounces() const { return m_RayTracingMaxBounces; }
    const RayTracingSceneBuffers* GetRayTracingScene() const { return m_RayTracingScene ? &*m_RayTracingScene : nullptr; }
    // While on, ray traced frames use the shader variant that counts its
    // traversal work. Slower, so benchmarks only switch it on after timing.
    void SetTraversalCounting(bool enabled) { m_CountTraversal = enabled; }
    // Totals since the last call; waits for the GPU to go idle
    TraversalCounts ReadAndResetTraversalCounters();
    // While on, ray traced frames use the shader variant that records every ray
    // it traces into capture queues (cleared when switched on). Uses ~0.8 GB.
    void SetRayCapture(bool enabled);
    // Traces the captured rays again with the trace-only kernel, each queue
    // `repetitions` times, and returns the median times. Waits for the GPU.
    std::vector<TraceOnlyQueueResult> RunTraceOnlyExperiment(uint32_t repetitions = 7);
    // Saves the latest ray traced frame (tonemapped) as a JPEG. Waits for the GPU to go idle.
    bool SaveRayTracingOutput(const std::filesystem::path& path);

    // For benchmark reports
    std::string GetDeviceName() const { return m_PhysicalDevice->getProperties().deviceName.data(); }
    vk::Extent2D GetSwapchainExtent() const { return m_SwapchainExtent; }

    AllocatedBuffer     CreateVertexBuffer(const std::vector<Momo::Geometry::Vertex>& vertices);
    AllocatedBuffer     CreateIndexBuffer(const std::vector<uint32_t>& indices);
    AllocatedImage      CreateAndSubmitTextureImage(const Assets::TextureData& texture);
    VulkanMaterialData  CreateVulkanMaterialData(const AllocatedImage& baseColorTexture, const MaterialParams& materialParams, bool doubleSided);
private:
    void CreateInstance();
    void PickPhysicalDevice();
    void CreateSurface();
    void CreateSwapChain();
    void RecreateSwapchain();
    void CreateLogicalDevice();
    void CreateImageView();
    void CreateGraphicsPipeline();
    void CreateCommandPool();
    void CreateCommandBuffers();
    void CreateSyncObjects();
    void CreateImageAvailableSemaphores();
    void CreateInFlightFences();
    void CreateRenderFinishedSemaphores();
    void CreateTextureSampler();
    void CreateTimestampQueryPool();
    void CheckTimestampSupport();
    void CreateDescriptorSetLayout();
    void WriteDescriptorSet(const vk::DescriptorSet descriptorSet, const AllocatedBuffer& ubo, const AllocatedImage& textureImage);

    void BeginFrame();
    void RecordRasterPass(uint32_t imageIndex, const VulkanModelData& modelData, const glm::mat4& viewMatrix, const glm::mat4& modelMatrix);
    // Sky behind the raster pass's geometry, and toon outlines over it (composite.slang)
    void RecordCompositePass(uint32_t imageIndex);
    // The scene's depth seen from the sun, for the raster pass's shadows (VulkanShadows.cpp)
    void RecordShadowPass(const VulkanModelData& modelData, const glm::mat4& modelMatrix);
    void CreateShadowMap();
    void CreateShadowPipeline();
    // World -> the shadow map's clip space: an orthographic box around the
    // scene bounds, looking along the sun's direction
    glm::mat4 SunViewProjection() const;
    // World-space size of one shadow map texel
    float ShadowTexelWorldSize() const;

    // Lighting shared by every pipeline (VulkanLighting.cpp)
    void CreateFrameResources();
    void WriteFrameUniforms(const glm::mat4& viewMatrix, RenderMode mode);
    void CreateCompositePipeline();
    AllocatedImage CreateNormalBuffer();
    void WriteCompositeDescriptorSet();
    // Draws the UI on top of the frame. Returns false if there was nothing to
    // draw, in which case the swapchain image's layout is left as it was.
    bool RecordUiPass(uint32_t imageIndex, RenderMode mode);
    // swapchainIsTransferDst: the ray tracing pass's blit was the last write
    void EndFrame(uint32_t imageIndex, bool swapchainIsTransferDst, RenderMode mode);

    // Ray tracing (VulkanRayTracing.cpp)
    void CheckRayTracingSupport();
    void CreateRayTracingPipeline();
    void CreateRayTracingImages();
    AllocatedImage CreateRayTracingStorageImage(vk::Format format, vk::ImageUsageFlags usage);
    void WriteRayTracingDescriptorSet();
    void RecordRayTracingPass(uint32_t imageIndex, const glm::mat4& viewMatrix);
    AllocatedBuffer CreateDeviceLocalBuffer(const void* data, vk::DeviceSize size, vk::BufferUsageFlags usage);

    static std::vector<char> ReadFile(const std::string& filename);
    vk::raii::ShaderModule CreateShaderModule(const std::vector<char>& code);
    uint32_t FindGraphicsQueueFamilyIdx(vk::raii::PhysicalDevice);
    uint32_t FindMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);

    AllocatedBuffer CreateBuffer(const void* data, vk::DeviceSize size, vk::BufferUsageFlags usage, uint32_t indexCount);
    AllocatedImage  CreateDepthBuffer();

    AllocatedBuffer CreateTextureStagingBuffer(const Assets::TextureData& texture);
    AllocatedImage  CreateTextureImage(const Assets::TextureData& texture);
    void AllocateDescriptorSet(const vk::raii::ImageView& imageView);
    void SubmitTextureImage(const AllocatedBuffer& stagingBuffer, const AllocatedImage& textureImage);

    vk::raii::CommandBuffer BeginSingleUseCommandBuffer();
    void EndSingleUseCommandBuffer(vk::raii::CommandBuffer& commandBuffer);

    const IWindow* m_Window = nullptr;
    vk::raii::Context  m_Context;
    std::optional<vk::raii::Instance> m_Instance;
    std::optional<vk::raii::SurfaceKHR> m_Surface;
    std::optional<vk::raii::PhysicalDevice> m_PhysicalDevice;
    std::optional<vk::raii::Device> m_Device;
    DescriptorAllocator m_DescriptorAllocator;

    // Swapchain details
    std::optional<vk::raii::SwapchainKHR> m_Swapchain;
    vk::SurfaceFormatKHR m_SwapchainImageFormat;
    vk::PresentModeKHR m_SwapchainPresentMode;
    vk::Extent2D m_SwapchainExtent;
    std::vector<vk::Image> m_SwapchainImages;
    std::vector<vk::raii::ImageView> m_SwapchainImageViews;

    // Graphics pipeline
    std::optional<vk::raii::PipelineLayout> m_PipelineLayout;
    std::optional<vk::raii::Pipeline> m_GraphicsPipeline;
    std::optional<vk::raii::QueryPool> m_TimestampQueryPool;

    // Command buffers (one per frame in flight)
    std::optional<vk::raii::CommandPool> m_CommandPool;
    std::vector<vk::raii::CommandBuffer> m_CommandBuffers;

    // Sync objects
    std::vector<vk::raii::Semaphore> m_ImageAvailableSemaphores;   // one per frame in flight
    std::vector<vk::raii::Semaphore> m_RenderFinishedSemaphores;   // one per swapchain image
    std::vector<vk::raii::Fence> m_InFlightFences;                 // one per frame in flight
    uint32_t m_CurrentFrame = 0;

    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_HasTimestampQueryPoolResults{};
    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_HasRayTracingTimestamps{}; // The slot's last frame wrote the dispatch pair
    uint64_t m_TimestampValidBitmask = 0;
    double m_LastGpuFrameMs = 0;
    std::optional<double> m_LastRayTracingMs;
    float m_TimestampPeriodNs = 0.0f;   // nanoseconds per timestamp tick
    bool  m_GpuTimingEnabled = false;   // false = skip every timestamp call later
    uint32_t m_GraphicsQueueFamilyIdx = 0;
    uint32_t m_PresentQueueFamilyIdx = 0;
    std::optional<vk::raii::Queue> m_GraphicsQueue;
    std::optional<vk::raii::Queue> m_PresentQueue;
    bool m_SubOptimal = false;

    // Depth buffer resources
    vk::Format m_DepthFormat;
    std::optional<AllocatedImage> m_DepthBuffer;

    // Texture Image resources
    std::optional<vk::raii::Sampler> m_TextureSampler;
    std::optional<vk::raii::DescriptorSetLayout> m_DescriptorSetLayout;

    // Lighting shared by every pipeline: set 1 of the raster, composite and
    // ray tracing pipelines. One uniform buffer and set per frame in flight, so
    // a frame's uniforms can be written while the previous frame still reads its own.
    SceneLighting m_Lighting;
    ToonSettings m_Toon;
    std::optional<vk::raii::DescriptorSetLayout> m_FrameSetLayout;
    DescriptorAllocator m_FrameDescriptorAllocator; // Also holds the composite set
    std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> m_FrameDescriptorSets{};
    std::vector<AllocatedBuffer> m_FrameUniformBuffers; // Host visible, mapped for the renderer's lifetime
    std::array<void*, MAX_FRAMES_IN_FLIGHT> m_FrameUniformsMapped{};
    std::optional<AllocatedBuffer> m_PointLightBuffer; // A zeroed placeholder entry when the scene has none
    uint32_t m_PointLightCount = 0;
    static constexpr float NearPlane = 0.1f;
    static constexpr float FarPlane = 100.0f;

    bool m_ImGuiInitialized = false;
    vk::Format m_ImGuiColorFormat; // Referenced by the backend's pipeline info

    // Sun shadow map: shared by all frames in flight, like the depth buffer
    static constexpr uint32_t ShadowMapSize = 4096;
    std::optional<AllocatedImage> m_ShadowMap;
    std::optional<vk::raii::Sampler> m_ShadowSampler; // Depth comparison
    std::optional<vk::raii::Pipeline> m_ShadowPipeline; // Uses the raster pass's pipeline layout
    glm::vec3 m_ShadowBoundsMin{ -1.0f };
    glm::vec3 m_ShadowBoundsMax{ 1.0f };

    // Composite pass: reads the raster pass's depth and normals
    std::optional<AllocatedImage> m_NormalBuffer; // World-space normals; a = 0 where nothing was drawn
    static constexpr vk::Format NormalBufferFormat = vk::Format::eR16G16B16A16Sfloat;
    std::optional<vk::raii::Sampler> m_NearestSampler;
    std::optional<vk::raii::DescriptorSetLayout> m_CompositeSetLayout;
    vk::DescriptorSet m_CompositeDescriptorSet;
    std::optional<vk::raii::PipelineLayout> m_CompositePipelineLayout;
    std::optional<vk::raii::Pipeline> m_CompositePipeline;

    // Ray tracing resources
    bool m_RayTracingSupported = false;
    bool m_SupportsNonUniformTextureIndexing = false; // Set at device creation
    std::optional<RayTracingSceneBuffers> m_RayTracingScene;
    std::optional<vk::raii::DescriptorSetLayout> m_RayTracingSetLayout;
    std::optional<vk::raii::PipelineLayout> m_RayTracingPipelineLayout;
    std::optional<vk::raii::Pipeline> m_RayTracingPipeline;
    std::optional<vk::raii::Pipeline> m_RayTracingCountingPipeline;
    std::optional<AllocatedBuffer> m_TraversalCounterBuffer; // (low, high) word pairs, host visible
    bool m_CountTraversal = false;

    // Trace-only experiment (raytrace_traversal.slang, traceonly.slang)
    static constexpr uint32_t CaptureBounceLimit = 6;
    static constexpr uint32_t CaptureQueueCount = 2 * CaptureBounceLimit;
    std::optional<vk::raii::Pipeline> m_RayTracingCapturePipeline;
    std::optional<vk::raii::PipelineLayout> m_TraceOnlyPipelineLayout;
    std::optional<vk::raii::Pipeline> m_TraceOnlyPipeline;
    std::optional<AllocatedBuffer> m_CapturedRays;   // CaptureQueueCount queues of m_CaptureCapacity rays
    std::optional<AllocatedBuffer> m_CaptureCounts;  // Host visible, one count per queue
    std::optional<AllocatedBuffer> m_TraceHits;
    uint32_t m_CaptureCapacity = 0;                  // Rays per queue: pixels per frame
    bool m_CaptureRays = false;
    void EnsureCaptureBuffers();
    AllocatedBuffer CreateUninitializedBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProperties);
    bool m_PipelineStatisticsEnabled = false; // VK_KHR_pipeline_executable_properties enabled
    void LogPipelineStatistics(const vk::raii::Pipeline& pipeline, const std::string& name);
    DescriptorAllocator m_RayTracingDescriptorAllocator;
    vk::DescriptorSet m_RayTracingDescriptorSet; // Allocated on first upload
    std::vector<vk::ImageView> m_RayTracingTextureViews; // Owned by the caller's texture cache
    // Shared by all frames in flight, like the depth buffer; sized to the swapchain
    std::optional<AllocatedImage> m_RayTracingOutput;
    static constexpr vk::Format RayTracingOutputFormat = vk::Format::eR16G16B16A16Sfloat;
    // Running average of every sample since the camera last moved. 32-bit:
    // averaging hundreds of samples in 16-bit floats loses precision
    std::optional<AllocatedImage> m_RayTracingAccumulation;
    static constexpr vk::Format RayTracingAccumulationFormat = vk::Format::eR32G32B32A32Sfloat;
    uint32_t m_AccumulatedSamples = 0; // 0 = the next frame starts a new average
    uint32_t m_RayTracingFrameIndex = 0;
    uint32_t m_RayTracingMaxBounces = 5;
    glm::mat4 m_AccumulationViewProjection{ 0.0f }; // Camera the running average was made with
    SceneLighting m_AccumulationLighting;            // Lighting it was made with

    // Intended for use when setting up Vulkan validation layers in instance creation.
    std::vector<char const*> m_ValidationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };

    std::vector<const char*> m_RequestedDeviceExtensions = {
        vk::KHRSwapchainExtensionName,
        vk::KHRDynamicRenderingExtensionName
    };

    std::vector<const char*> m_OptionalDeviceExtensions = {
        "VK_KHR_portability_subset", // For MoltenVK on macOS
        "VK_KHR_pipeline_executable_properties", // Compiled shader statistics (registers, ...), logged for ray tracing
    };

    std::vector<const char*> m_EnabledDeviceExtensions;

    glm::mat4 m_ProjectionMatrix;

    // Debug Timers
    Profiling::Timer m_RecordTimer, m_SubmitTimer, m_FenceWaitTimer;
};
} // namespace Renderer
} // namespace Momo
