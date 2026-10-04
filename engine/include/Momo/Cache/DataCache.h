#pragma once

#include "Momo/Assets/Handle.h"
#include "Momo/Assets/AssetRegistry.h"
#include "Momo/Renderer/VulkanRenderer.h"
#include <unordered_map>

namespace Momo::Renderer {

template <typename Tag> class GPUCache;
template <typename Tag> struct GPUResourceTraits;

struct GPUContext {
    VulkanRenderer& renderer;
    GPUCache<Assets::TextureTag>* textures = nullptr; // pointer because the texture cache's own context has none
};

template <typename Tag>
class GPUCache
{
public:
    using GPUType = typename GPUResourceTraits<Tag>::GPUType;

    GPUCache(const Assets::AssetRegistry& assetRegistry, GPUContext ctx) 
        : m_AssetRegistry(assetRegistry), m_Context(ctx) {}

    ~GPUCache() = default;

    const GPUType& GetOrCreate(Assets::Handle<Tag> handle) {
        auto it = m_Cache.find(handle);
        if (it == m_Cache.end()) {
            it = m_Cache.emplace(handle, GPUResourceTraits<Tag>::Create(m_AssetRegistry.Get(handle), m_Context)).first;
        }
        return it->second;
    }
private:
    std::unordered_map<Assets::Handle<Tag>, GPUType> m_Cache;
    const Assets::AssetRegistry& m_AssetRegistry;
    GPUContext m_Context;
};

using TextureGPUCache = GPUCache<Assets::TextureTag>;
using MaterialGPUCache = GPUCache<Assets::MaterialTag>;
using MeshGPUCache = GPUCache<Assets::MeshTag>;


template <> struct GPUResourceTraits<Assets::TextureTag> {
    using GPUType = AllocatedImage;
    static GPUType Create(const Assets::TextureData& textureData, GPUContext ctx) {
        return ctx.renderer.CreateAndSubmitTextureImage(textureData);
    }
};

template <> struct GPUResourceTraits<Assets::MeshTag> {
    using GPUType = GPUMesh;
    static GPUType Create(const Assets::Mesh& mesh, GPUContext ctx) {
        return GPUMesh {
                    .vertexBuffer = ctx.renderer.CreateVertexBuffer(mesh.meshData.vertices),
                    .indexBuffer = ctx.renderer.CreateIndexBuffer(mesh.meshData.indices),
                };
    }
};

template <> struct GPUResourceTraits<Assets::MaterialTag> {
    using GPUType = VulkanMaterialData;
    static GPUType Create(const Assets::Material& material, GPUContext ctx) {
        return VulkanMaterialData {
            .doubleSided = material.doubleSided,
            .baseColorTextureDescriptorSet = ctx.renderer.CreateMaterialDescriptorSet(
                ctx.textures->GetOrCreate(material.baseColorTextureHandle)
            )
        };
    }
};


} // namespace Momo
