#include "Momo/RayTracing/SceneGeometry.h"

#include <cmath>

namespace Momo::RayTracing {
namespace {
// Zero-length normals in source data would normalize to NaN and poison shading
glm::vec3 NormalizeOr(const glm::vec3& normal, const glm::vec3& fallback)
{
    float lengthSquared = glm::dot(normal, normal);
    return lengthSquared > 0.0f ? normal / std::sqrt(lengthSquared) : fallback;
}
} // namespace

SceneGeometry BuildSceneGeometry(const Assets::AssetRegistry& registry, Assets::ModelHandle model,
                                 const glm::mat4& modelMatrix, const BvhBuildSettings& settings)
{
    std::vector<TrianglePositions> positions;
    std::vector<TriangleShading> shading;

    for (const Assets::MeshHandle& meshHandle : registry.Get(model).meshes)
    {
        const Assets::Mesh& mesh = registry.Get(meshHandle);
        const auto& vertices = mesh.meshData.vertices;
        const auto& indices = mesh.meshData.indices;

        glm::mat4 toWorld = modelMatrix * mesh.localTransform;
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
                .materialIndex = mesh.materialHandle.id,
            });
        }
    }

    SceneGeometry geometry;
    geometry.bvh = BuildBvh(positions, settings);
    geometry.positions = Reorder<TrianglePositions>(positions, geometry.bvh.triangleOrder);
    geometry.shading = Reorder<TriangleShading>(shading, geometry.bvh.triangleOrder);
    return geometry;
}

} // namespace Momo::RayTracing
