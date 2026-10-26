#pragma once
#include "Momo/Geometry/Vertex.h"
#include "Handle.h"

namespace Momo::Assets {
struct TextureData {
    std::vector<uint8_t> pixelData; // Raw pixel data of the texture
    uint32_t width = 0;  // Width of the texture
    uint32_t height = 0; // Height of the texture
    uint32_t channels = 0; // Number of color channels in the texture (e.g., 3 for RGB, 4 for RGBA)

    // TODO: Support sampling parameters (e.g., filtering, wrapping)
    // Filtering mode (e.g., nearest, linear)
    // Wrapping mode (e.g., repeat, clamp to edge)
};

enum class AlphaMode {
    Opaque,
    Mask,
    Blend
};

struct Material {
    glm::vec4 baseColorFactor = glm::vec4(0.0f);
    bool doubleSided = false; // Whether the material is double-sided
    float alphaCutoff = 0.5f; // Cutoff value for alpha masking
    AlphaMode alphaMode = AlphaMode::Opaque; // Alpha mode for the material
    TextureHandle baseColorTextureHandle; // Handle to the base color texture for the material
    glm::vec3 emissive = glm::vec3(0.0f); // Light the surface gives off: emissiveFactor * emissive strength
};

// A point light (KHR_lights_punctual) placed in the model
struct PointLight {
    glm::vec3 position;
    // Radiant intensity per channel (color * intensity): a surface at distance d
    // facing the light receives intensity / d^2
    glm::vec3 intensity;
};

struct Mesh {
    Geometry::MeshData meshData;
    glm::mat4 localTransform = glm::mat4(1.0f);
    MaterialHandle materialHandle; // Handle to the material used by this mesh
};

// A registered model: the meshes it owns, addressed by handle so several
// models can share the same mesh, material or texture.
struct Model {
    std::vector<MeshHandle> meshes;
    std::vector<PointLight> pointLights;
};
} // namespace Momo::Assets
