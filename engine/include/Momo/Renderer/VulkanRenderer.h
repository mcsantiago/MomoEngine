#pragma once
#include <vulkan/vulkan_raii.hpp>
#include <vector>
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

namespace Momo {
namespace Renderer {
struct DrawFrameTime
{
    float recordMs, submitMs, fenceWaitMs, gpuFrameMs;
};

enum class RenderMode
{
    Raster,
    RayTraced
};

// Push constants of raytrace.slang: enough to rebuild each pixel's camera ray
struct RayTracingPushConstants
{
    glm::mat4 inverseView;
    glm::mat4 inverseProjection;
};
static_assert(sizeof(RayTracingPushConstants) == 128, "RayTracingPushConstants must fit the guaranteed 128 bytes");

// GPU copies of RayTracing::SceneGeometry, in device-local storage buffers
struct RayTracingSceneBuffers
{
    AllocatedBuffer nodes;
    AllocatedBuffer positions;
    AllocatedBuffer shading;
    AllocatedBuffer materials;
    RayTracing::BvhStats stats;
};

class VulkanRenderer
{
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    void Init(const IWindow& window);
    void Shutdown();
    DrawFrameTime RenderFrame(Renderer::VulkanModelData& modelData, glm::mat4 viewMatrix, glm::mat4 modelMatrix, RenderMode mode);

    // Uploads the scene's ray tracing geometry and keeps only the GPU copy.
    // textureViews[i] is the GPU image of geometry.textures[i]; the caller keeps
    // the images alive while the scene is in use. Replaces any previous scene.
    void UploadRayTracingScene(RayTracing::SceneGeometry geometry, std::vector<vk::ImageView> textureViews);
    // False if the device can't present a ray traced image or no scene was uploaded
    bool IsRayTracingAvailable() const { return m_RayTracingSupported && m_RayTracingScene.has_value(); }

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
    void EndFrame(uint32_t imageIndex, RenderMode mode);

    // Ray tracing (VulkanRayTracing.cpp)
    void CheckRayTracingSupport();
    void CreateRayTracingPipeline();
    void CreateRayTracingOutputImage();
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
    uint64_t m_TimestampValidBitmask = 0;
    double m_LastGpuFrameMs = 0;
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

    // Ray tracing resources
    bool m_RayTracingSupported = false;
    bool m_SupportsNonUniformTextureIndexing = false; // Set at device creation
    std::optional<RayTracingSceneBuffers> m_RayTracingScene;
    std::optional<vk::raii::DescriptorSetLayout> m_RayTracingSetLayout;
    std::optional<vk::raii::PipelineLayout> m_RayTracingPipelineLayout;
    std::optional<vk::raii::Pipeline> m_RayTracingPipeline;
    DescriptorAllocator m_RayTracingDescriptorAllocator;
    vk::DescriptorSet m_RayTracingDescriptorSet; // Allocated on first upload
    std::vector<vk::ImageView> m_RayTracingTextureViews; // Owned by the caller's texture cache
    // Shared by all frames in flight, like the depth buffer; sized to the swapchain
    std::optional<AllocatedImage> m_RayTracingOutput;
    static constexpr vk::Format RayTracingOutputFormat = vk::Format::eR16G16B16A16Sfloat;

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
    };

    std::vector<const char*> m_EnabledDeviceExtensions;

    glm::mat4 m_ProjectionMatrix;

    // Debug Timers
    Profiling::Timer m_RecordTimer, m_SubmitTimer, m_FenceWaitTimer;
};
} // namespace Renderer
} // namespace Momo
