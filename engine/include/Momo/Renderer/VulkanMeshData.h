#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cstdint>
#include <optional>
#include "Momo/Assets/Handle.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // Vulkan depth [0, 1] range
#include <glm/glm.hpp>

namespace Momo {
namespace Renderer {
struct alignas(16) MaterialParams
{
    glm::vec4 baseColorFactor;
    float alphaCutoff;
    float _pad[3]; // Padding to make the struct size a multiple of 16 bytes
};
static_assert(sizeof(MaterialParams) == 32, "MaterialParams size must be 32 bytes");

struct PushConstantData
{
    glm::mat4 projectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;
};
static_assert(sizeof(PushConstantData) == 192, "PushConstantData size must be 192 bytes");

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
    vk::DescriptorSet descriptorSet; 
    AllocatedBuffer paramsBuffer; // UBO for the material's parameters (baseColorFactor, alphaCutoff)
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
    const VulkanMaterialData& materialData; // Descriptor set for the mesh's material
};

struct VulkanModelData
{
    std::vector<VulkanMeshData> meshes;
    // std::optional<AllocatedImage> textureImage;
};
} // namespace Renderer
} // namespace Momo
