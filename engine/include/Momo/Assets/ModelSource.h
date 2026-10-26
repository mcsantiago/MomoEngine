#pragma once

#include "Momo/Geometry/Vertex.h"
#include "Momo/Assets/ModelData.h"
#include <vector>
#include <optional>
#include <cstdint>
#include <glm/glm.hpp>

/*
 * Source data classes from ModelLoader to be registered in the Registry
 **/
namespace Momo::Assets {
struct TextureSource {
    // Raw, tightly packed pixels: width * height * channels bytes.
    std::vector<uint8_t> pixels;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;

    bool IsEmpty() const { return pixels.empty(); }
};

struct MaterialSource {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    // Index into ModelSource::textures, which mirrors the glTF image list.
    bool doubleSided = false;
    std::optional<uint32_t> baseColorTexture;
    AlphaMode alphaMode = AlphaMode::Opaque;
    float alphaCutoff = 0.5f;
    glm::vec3 emissive = glm::vec3(0.0f);
};

struct MeshSource {
    Geometry::MeshData geometry;
    glm::mat4 localTransform = glm::mat4(1.0f);
    // Index into ModelSource::materials, which mirrors the glTF material list.
    std::optional<uint32_t> materialIndex;
};

// A camera placed in the model (the first one in the file)
struct CameraSource {
    glm::vec3 position;
    glm::vec3 forward; // Unit length
};

struct ModelSource {
    // meshes is one entry per drawn primitive; materials and textures are flat
    // file-level tables whose indices match the glTF material / image lists, so
    // primitives that share a material or image share the entry here too.
    std::vector<MeshSource> meshes;
    std::vector<MaterialSource> materials;
    std::vector<TextureSource> textures;
    // In model space, like MeshSource::localTransform
    std::vector<PointLight> pointLights;
    std::optional<CameraSource> camera;
};
} // namespace Momo::Assets
