#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <optional>
#include "Momo/Assets/Handle.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Vulkan depth [0, 1] range
#include <glm/glm.hpp>

namespace Momo {
namespace Renderer {
struct PushConstantData
{
    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;
    glm::vec4 baseColorFactor;
};

struct AllocatedImage
{
    vk::raii::DeviceMemory memory;
    vk::raii::Image image;
    vk::raii::ImageView imageView;
    uint32_t width;
    uint32_t height;
};

struct AllocatedBuffer
{
    vk::raii::DeviceMemory memory;
    vk::raii::Buffer buffer;
    uint32_t indexCount;
};

struct VulkanTextureData
{
    std::optional<AllocatedImage> textureImage;
    Assets::TextureHandle textureHandle;
};

struct VulkanMaterialData
{
    bool doubleSided;
    vk::DescriptorSet baseColorTextureDescriptorSet; 
};

struct GPUMesh
{
    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;
};

struct VulkanMeshData
{
    const GPUMesh& gpuMesh;
    glm::mat4 localTransform;
    glm::vec4 baseColorFactor;
    VulkanMaterialData materialData; // Descriptor set for the mesh's material
};

struct VulkanModelData
{
    std::vector<VulkanMeshData> meshes;
    // std::optional<AllocatedImage> textureImage;
};
} // namespace Renderer
} // namespace Momo
