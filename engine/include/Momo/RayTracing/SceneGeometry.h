#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "Momo/Assets/AssetRegistry.h"
#include "Momo/Core/Scene.h"
#include "Momo/RayTracing/Bvh.h"
#include "Momo/RayTracing/WideBvh.h"

namespace Momo::RayTracing {

// Size of the shader's texture array (raytrace.slang must use the same value).
// A scene using more textures gets white for the extras.
inline constexpr uint32_t MaxRayTracingTextures = 256;

// Per-material data for the compute shader. TriangleShading::materialIndex
// indexes a compact table of these, holding only the scene's materials.
struct RayTracingMaterial {
    glm::vec4 baseColorFactor;
    glm::vec4 emissive;         // rgb: light the surface gives off; a unused
    float     alphaCutoff;      // Mask materials: the cutoff; Opaque and Blend: -1 so nothing is discarded
    uint32_t  baseColorTexture; // Index into SceneGeometry::textures
    uint32_t  padding[2] = { 0, 0 };
};
static_assert(sizeof(RayTracingMaterial) == 48, "RayTracingMaterial must match the std430 layout in the shader");

// A point light in world space, as the compute shader reads it
struct RayTracingPointLight {
    glm::vec4 position;  // w unused
    glm::vec4 intensity; // rgb: radiant intensity (see Assets::PointLight); a unused
};
static_assert(sizeof(RayTracingPointLight) == 32, "RayTracingPointLight must match the std430 layout in the shader");

// Everything the ray tracing compute shader reads about the scene's geometry,
// already in the layout it is uploaded in: one world-space triangle list for
// the whole scene (single-level compressed wide BVH) with both triangle arrays in BVH order.
// Built on the CPU, then handed to the renderer, which uploads it and drops it.
struct SceneGeometry {
    WideBvh bvh;
    std::vector<TrianglePositions> positions;
    std::vector<TriangleShading> shading;
    std::vector<RayTracingMaterial> materials;
    // Registry textures the materials sample, by slot; slot 0 is the default
    // white texture. The renderer binds their GPU images as an array.
    std::vector<Assets::TextureHandle> textures;
    std::vector<RayTracingPointLight> pointLights;
};

// Flattens every mesh of every model in the scene into world space (the mesh's
// localTransform; models have no transform of their own yet) and builds the
// BVH over the result. Meshes without a material get a default white one.
SceneGeometry BuildSceneGeometry(const Assets::AssetRegistry& registry, const Scene& scene);

} // namespace Momo::RayTracing
