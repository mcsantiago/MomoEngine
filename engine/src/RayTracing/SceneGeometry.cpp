#include "Momo/RayTracing/SceneGeometry.h"

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

// Mirrors the raster path's MaterialParams (see MaterialGPUCache)
RayTracingMaterial ToRayTracingMaterial(const Assets::AssetRegistry& registry, const Assets::Material& material)
{
    return RayTracingMaterial{
        .baseColorFactor = material.baseColorFactor,
        .alphaCutoff = material.alphaMode == Assets::AlphaMode::Mask ? material.alphaCutoff : -1.0f,
        .baseColorTexture = registry.Resolve(material.baseColorTextureHandle).id,
    };
}

// Maps registry material handles to indices in a compact table holding only
// the materials the scene uses, in first-use order
class MaterialTable
{
public:
    explicit MaterialTable(const Assets::AssetRegistry& registry) : m_Registry(registry) {}

    uint32_t IndexOf(Assets::MaterialHandle handle)
    {
        auto [it, inserted] = m_Indices.try_emplace(handle, static_cast<uint32_t>(m_Materials.size()));
        if (inserted)
        {
            m_Materials.push_back(handle.IsValid()
                ? ToRayTracingMaterial(m_Registry, m_Registry.Get(handle))
                : RayTracingMaterial{ .baseColorFactor = glm::vec4(1.0f), .alphaCutoff = -1.0f,
                                      .baseColorTexture = m_Registry.Resolve(Assets::TextureHandle{}).id });
        }
        return it->second;
    }

    std::vector<RayTracingMaterial> Take() { return std::move(m_Materials); }

private:
    const Assets::AssetRegistry& m_Registry;
    std::unordered_map<Assets::MaterialHandle, uint32_t> m_Indices;
    std::vector<RayTracingMaterial> m_Materials;
};

// Appends the mesh's triangles in world space
void AppendMesh(const Assets::Mesh& mesh, uint32_t materialIndex,
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
            .v0 = glm::vec4(p0, 0.0f),
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
            AppendMesh(mesh, materials.IndexOf(mesh.materialHandle), positions, shading);
        }
    }

    SceneGeometry geometry;
    geometry.bvh = BuildBvh(positions, settings);
    geometry.positions = Reorder<TrianglePositions>(positions, geometry.bvh.triangleOrder);
    geometry.shading = Reorder<TriangleShading>(shading, geometry.bvh.triangleOrder);
    geometry.materials = materials.Take();
    return geometry;
}

} // namespace Momo::RayTracing
