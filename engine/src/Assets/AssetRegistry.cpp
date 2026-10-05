#include "Momo/Assets/AssetRegistry.h"
#include "Momo/Logging/Logger.h"

#include <utility>

namespace Momo::Assets {
namespace {
TextureData ToTextureData(TextureSource&& source) {
    TextureData textureData;
    textureData.pixelData = std::move(source.pixels);
    textureData.width = source.width;
    textureData.height = source.height;
    textureData.channels = source.channels;
    return textureData;
}

Material ToMaterial(const MaterialSource& source, const TextureHandle textureHandle) {
    Material material;
    material.doubleSided = source.doubleSided;
    material.baseColorFactor = source.baseColorFactor;
    material.alphaMode = source.alphaMode;
    material.alphaCutoff = source.alphaCutoff;
    material.baseColorTextureHandle = textureHandle;
    return material;
}

Mesh ToMesh(MeshSource&& source, const MaterialHandle materialHandle) {
    Mesh mesh;
    mesh.meshData = std::move(source.geometry);
    mesh.localTransform = source.localTransform;
    mesh.materialHandle = materialHandle;
    return mesh;
}
} // namespace

void AssetRegistry::Init() {
    TextureData white;
    white.pixelData = {255, 255, 255, 255};
    white.width = 1;
    white.height = 1;
    white.channels = 4;
    m_DefaultTexture = m_TextureAssets.Add(std::move(white));

    // 64x64 with 8x8 cells rather than 2x2: the shared sampler filters
    // linearly, so a tiny texture stretched over a mesh would blur into a
    // purple smear. Larger cells keep the blur to a thin band at each edge.
    constexpr uint32_t size = 64;
    constexpr uint32_t cell = 8;
    TextureData checker;
    checker.width = size;
    checker.height = size;
    checker.channels = 4;
    checker.pixelData.resize(size * size * 4);
    for (uint32_t y = 0; y < size; ++y) {
        for (uint32_t x = 0; x < size; ++x) {
            const bool magenta = ((x / cell) + (y / cell)) % 2 == 0;
            uint8_t* pixel = &checker.pixelData[(y * size + x) * 4];
            pixel[0] = magenta ? 255 : 0;
            pixel[1] = 0;
            pixel[2] = magenta ? 255 : 0;
            pixel[3] = 255;
        }
    }
    m_MissingTexture = m_TextureAssets.Add(std::move(checker));
}

TextureHandle AssetRegistry::Resolve(TextureHandle handle) const {
    if (m_TextureAssets.Has(handle)) {
        return handle;
    }
    // An Invalid handle just means "no texture", which glTF allows. A valid id
    // that isn't in the pool means something removed it out from under us.
    if (handle.IsValid()) {
        LOG_WARN("AssetRegistry", "Texture handle {} not found. Using m_Missing texture.", handle.id);
        return m_MissingTexture;
    }
    return m_DefaultTexture;
}

ModelHandle AssetRegistry::RegisterModel(ModelSource&& source) {
    // Local index -> handle lookup tables. Slots the loader left empty stay
    // invalid, so a reference to a failed decode is detectable rather than
    // silently pointing at the wrong asset.
    std::vector<TextureHandle> textureHandles(source.textures.size());
    for (size_t i = 0; i < source.textures.size(); ++i) {
        if (!source.textures[i].IsEmpty()) {
            textureHandles[i] = m_TextureAssets.Add(ToTextureData(std::move(source.textures[i])));
        }
    }

    std::vector<MaterialHandle> materialHandles(source.materials.size());
    for (size_t i = 0; i < source.materials.size(); ++i) {
        const auto &src = source.materials[i];
        TextureHandle textureHandle{};
        if (src.baseColorTexture) {
            textureHandle = textureHandles[*src.baseColorTexture];
            if (!textureHandle.IsValid()) {
                LOG_WARN("AssetRegistry", "Material {} references texture {} which failed to decode. Using m_Missing texture.", i, *src.baseColorTexture);
                textureHandle = m_MissingTexture;
            }
        }
        materialHandles[i] = m_MaterialAssets.Add(ToMaterial(src, Resolve(textureHandle)));
    }

    Model model;
    model.meshes.reserve(source.meshes.size());
    for (auto& meshSource : source.meshes) {
        // TODO: Consider using a default material or logging a warning when a mesh has no material.
        const auto materialHandle = meshSource.materialIndex ? materialHandles[*meshSource.materialIndex] : MaterialHandle{};
        Mesh mesh = ToMesh(std::move(meshSource), materialHandle);
        model.meshes.push_back(m_MeshAssets.Add(std::move(mesh)));
    }

    return m_ModelAssets.Add(std::move(model));
}
} // namespace Momo::Assets
