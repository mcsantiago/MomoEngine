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

namespace Momo {
namespace Renderer {
struct DrawFrameTime
{
    float recordMs, submitMs, fenceWaitMs, gpuFrameMs;
};

class VulkanRenderer
{
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    void Init(const IWindow& window);
    void Shutdown();
    DrawFrameTime RenderFrame(Renderer::VulkanModelData& modelData, glm::mat4 viewMatrix, glm::mat4 modelMatrix);

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

    void BeginFrame(uint32_t imageIndex);
    void EndFrame(uint32_t imageIndex);
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
