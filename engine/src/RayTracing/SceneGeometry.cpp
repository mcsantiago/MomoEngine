#include "Momo/RayTracing/SceneGeometry.h"
#include "Momo/Logging/Logger.h"

#include <cmath>
#include <unordered_map>

namespace Momo::RayTracing {
namespace {
// Zero-length normals in source data would normalize to NaN and poison shading
glm::vec3 NormalizeOr(const glm::vec3& normal, const glm::vec3& fallback)
{
    float lengthSquared = glm::dot(normal, normal);
    return lengthSquared > 0.0f ? normal / std::sqrt(lengthSquared) : fallback;
}

// Maps registry material and texture handles to indices in compact tables
// holding only what the scene uses, in first-use order. Texture slot 0 is
// always the default white texture.
class MaterialTable
{
public:
    explicit MaterialTable(const Assets::AssetRegistry& registry) : m_Registry(registry)
    {
        TextureIndexOf(m_Registry.Resolve(Assets::TextureHandle{}));
    }

    uint32_t IndexOf(Assets::MaterialHandle handle)
    {
        auto [it, inserted] = m_MaterialIndices.try_emplace(handle, static_cast<uint32_t>(m_Materials.size()));
        if (inserted)
        {
            // Meshes without a material get an opaque white one
            Assets::Material material{ .baseColorFactor = glm::vec4(1.0f) };
            if (handle.IsValid())
                material = m_Registry.Get(handle);

            // Mirrors the raster path's MaterialParams (see MaterialGPUCache)
            m_Materials.push_back(RayTracingMaterial{
                .baseColorFactor = material.baseColorFactor,
                .alphaCutoff = material.alphaMode == Assets::AlphaMode::Mask ? material.alphaCutoff : -1.0f,
                .baseColorTexture = TextureIndexOf(m_Registry.Resolve(material.baseColorTextureHandle)),
            });
        }
        return it->second;
    }

    const RayTracingMaterial& Get(uint32_t index) const { return m_Materials[index]; }

    std::vector<RayTracingMaterial> TakeMaterials() { return std::move(m_Materials); }
    std::vector<Assets::TextureHandle> TakeTextures() { return std::move(m_Textures); }

private:
    uint32_t TextureIndexOf(Assets::TextureHandle handle)
    {
        auto it = m_TextureIndices.find(handle);
        if (it != m_TextureIndices.end())
            return it->second;

        if (m_Textures.size() == MaxRayTracingTextures)
        {
            LOG_WARN("RayTracing", "Scene uses more than {} textures; texture {} falls back to white", MaxRayTracingTextures, handle.id);
            return 0;
        }
        auto index = static_cast<uint32_t>(m_Textures.size());
        m_Textures.push_back(handle);
        m_TextureIndices.emplace(handle, index);
        return index;
    }

    const Assets::AssetRegistry& m_Registry;
    std::unordered_map<Assets::MaterialHandle, uint32_t> m_MaterialIndices;
    std::vector<RayTracingMaterial> m_Materials;
    std::unordered_map<Assets::TextureHandle, uint32_t> m_TextureIndices;
    std::vector<Assets::TextureHandle> m_Textures;
};

// Appends the mesh's triangles in world space
void AppendMesh(const Assets::Mesh& mesh, uint32_t materialIndex, bool alphaMasked,
                std::vector<TrianglePositions>& positions, std::vector<TriangleShading>& shading)
{
    const auto& vertices = mesh.meshData.vertices;
    const auto& indices = mesh.meshData.indices;

    const glm::mat4& toWorld = mesh.localTransform;
    // Normals transform by the inverse transpose so non-uniform scale doesn't skew them
    glm::mat3 normalToWorld = glm::transpose(glm::inverse(glm::mat3(toWorld)));

    positions.reserve(positions.size() + indices.size() / 3);
    shading.reserve(shading.size() + indices.size() / 3);
    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const Geometry::Vertex& a = vertices[indices[i]];
        const Geometry::Vertex& b = vertices[indices[i + 1]];
        const Geometry::Vertex& c = vertices[indices[i + 2]];

        glm::vec3 p0 = glm::vec3(toWorld * glm::vec4(a.position, 1.0f));
        glm::vec3 p1 = glm::vec3(toWorld * glm::vec4(b.position, 1.0f));
        glm::vec3 p2 = glm::vec3(toWorld * glm::vec4(c.position, 1.0f));
        glm::vec3 faceNormal = NormalizeOr(glm::cross(p1 - p0, p2 - p0), glm::vec3(0.0f, 1.0f, 0.0f));

        positions.push_back(TrianglePositions{
            .v0 = glm::vec4(p0, alphaMasked ? 1.0f : 0.0f),
            .v1 = glm::vec4(p1, 0.0f),
            .v2 = glm::vec4(p2, 0.0f),
        });
        shading.push_back(TriangleShading{
            .normal0 = glm::vec4(NormalizeOr(normalToWorld * a.normal, faceNormal), 0.0f),
            .normal1 = glm::vec4(NormalizeOr(normalToWorld * b.normal, faceNormal), 0.0f),
            .normal2 = glm::vec4(NormalizeOr(normalToWorld * c.normal, faceNormal), 0.0f),
            .uv0 = a.texCoord,
            .uv1 = b.texCoord,
            .uv2 = c.texCoord,
            .materialIndex = materialIndex,
        });
    }
}
} // namespace

SceneGeometry BuildSceneGeometry(const Assets::AssetRegistry& registry, const Scene& scene, const BvhBuildSettings& settings)
{
    std::vector<TrianglePositions> positions;
    std::vector<TriangleShading> shading;
    MaterialTable materials(registry);

    for (const Assets::ModelHandle& model : scene.GetModels())
    {
        for (const Assets::MeshHandle& meshHandle : registry.Get(model).meshes)
        {
            const Assets::Mesh& mesh = registry.Get(meshHandle);
            uint32_t materialIndex = materials.IndexOf(mesh.materialHandle);
            bool alphaMasked = materials.Get(materialIndex).alphaCutoff >= 0.0f;
            AppendMesh(mesh, materialIndex, alphaMasked, positions, shading);
        }
    }

    SceneGeometry geometry;
    geometry.bvh = BuildBvh(positions, settings);
    geometry.positions = Reorder<TrianglePositions>(positions, geometry.bvh.triangleOrder);
    geometry.shading = Reorder<TriangleShading>(shading, geometry.bvh.triangleOrder);
    geometry.materials = materials.TakeMaterials();
    geometry.textures = materials.TakeTextures();
    return geometry;
}

} // namespace Momo::RayTracing
