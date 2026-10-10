#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Momo/Assets/AssetRegistry.h"
#include "Momo/RayTracing/Bvh.h"

namespace Momo::RayTracing {

// Everything the ray tracing compute shader reads about the scene's geometry,
// already in the layout it is uploaded in: one world-space triangle list for
// the whole scene (single-level BVH) with both triangle arrays in BVH order.
struct SceneGeometry {
    Bvh bvh;
    std::vector<TrianglePositions> positions;
    std::vector<TriangleShading> shading;
};

// Flattens every mesh of the model into world space (modelMatrix * the mesh's
// localTransform) and builds the BVH over the result. materialIndex is the
// material's registry handle id, so the GPU can index registry-wide material data.
SceneGeometry BuildSceneGeometry(const Assets::AssetRegistry& registry, Assets::ModelHandle model,
                                 const glm::mat4& modelMatrix = glm::mat4(1.0f),
                                 const BvhBuildSettings& settings = {});

} // namespace Momo::RayTracing
