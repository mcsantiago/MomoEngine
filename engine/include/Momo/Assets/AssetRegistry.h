#pragma once

#include "AssetPool.h"
#include "Handle.h"
#include "ModelData.h"
#include "ModelSource.h"

#include <unordered_map>
#include <string>

namespace Momo::Assets {
class AssetRegistry
{
public:
    const TextureData& Get(TextureHandle handle) const { return m_TextureAssets.Get(handle); }

    // TODO: Handle default Material for missing materials
    const Material& Get(MaterialHandle handle) const { return m_MaterialAssets.Get(handle); }

    // Missing meshes is genuinely an error and should be handled appropriately.
    const Mesh& Get(MeshHandle handle) const { return m_MeshAssets.Get(handle); }
    const Model& Get(ModelHandle handle) const { return m_ModelAssets.Get(handle); }

    // Registers everything a loader produced and hands back the model. This is
    // the only place local ModelSource indices become engine-wide handles:
    // textures first, then materials that reference them, then meshes.
    // Consumes the source: pixel and geometry buffers are moved into the pools.
    ModelHandle RegisterModel(ModelSource&& source);

    // Registers engine-owned fallback assets. Must run before RegisterModel.
    void Init();

    // Returns the handle unchanged if it names a live texture. Otherwise an
    // Invalid handle ("no texture") resolves to the default white texture and
    // a valid-but-missing one to the missing-texture checkerboard. Resolving
    // here, rather than inside Get, means every fallback shares one handle, so
    // GPUCache uploads each fallback once.
    TextureHandle Resolve(TextureHandle handle) const;

private:
    AssetPool<TextureData, TextureTag> m_TextureAssets;
    AssetPool<Material, MaterialTag> m_MaterialAssets;
    AssetPool<Mesh, MeshTag> m_MeshAssets;
    AssetPool<Model, ModelTag> m_ModelAssets;

    // 1x1 opaque white: multiplies out to baseColorFactor in the shader.
    TextureHandle m_DefaultTexture;
    // Magenta/black checkerboard for textures that should exist but don't, so
    // broken assets are obvious on screen rather than only in the log.
    TextureHandle m_MissingTexture;

    std::unordered_map<std::string, TextureHandle> m_TexturesByPath;
};

} // namespace Momo::Assets
