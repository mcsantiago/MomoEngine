#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <glm/glm.hpp>

namespace Momo::RayTracing {

// GPU-facing layouts. These are uploaded as-is into storage buffers and read
// by the compute shader with std430 rules, so every vec3 is either padded to a
// vec4 or followed by a 4-byte scalar. The static_asserts catch drift.

// One node of the flattened BVH. Interior nodes have triangleCount == 0 and
// leftOrFirst = index of the left child; the right child is always
// leftOrFirst + 1. Leaves have triangleCount > 0 and leftOrFirst = index of
// their first triangle, with the leaf's triangles stored contiguously.
struct BvhNode {
    glm::vec3 boundsMin;
    uint32_t  leftOrFirst;
    glm::vec3 boundsMax;
    uint32_t  triangleCount;

    bool IsLeaf() const { return triangleCount > 0; }
};
static_assert(sizeof(BvhNode) == 32, "BvhNode must match the std430 layout in the shader");

// World-space triangle positions, the only data traversal touches. Kept apart
// from shading data so the hot loop reads 48 bytes per candidate triangle.
struct TrianglePositions {
    glm::vec4 v0; // w unused
    glm::vec4 v1; // w unused
    glm::vec4 v2; // w unused
};
static_assert(sizeof(TrianglePositions) == 48, "TrianglePositions must match the std430 layout in the shader");

// Per-triangle shading data, read only once the closest hit is known.
struct TriangleShading {
    glm::vec4 normal0; // w unused
    glm::vec4 normal1; // w unused
    glm::vec4 normal2; // w unused
    glm::vec2 uv0;
    glm::vec2 uv1;
    glm::vec2 uv2;
    uint32_t  materialIndex;
    uint32_t  padding = 0;
};
static_assert(sizeof(TriangleShading) == 80, "TriangleShading must match the std430 layout in the shader");

struct BvhBuildSettings {
    uint32_t binCount = 16;          // SAH candidate splits per axis
    uint32_t maxLeafTriangles = 8;   // Leaves larger than this are split even if SAH prefers a leaf
    uint32_t maxDepth = 32;          // Bounds the GPU traversal stack; deeper nodes become leaves
    float    traversalCost = 1.0f;   // SAH cost of visiting an interior node
    float    intersectionCost = 1.0f; // SAH cost of one ray/triangle test
};

struct BvhStats {
    uint32_t nodeCount = 0;
    uint32_t leafCount = 0;
    uint32_t maxDepth = 0;
    uint32_t maxLeafTriangles = 0;
    float    sahCost = 0.0f; // Expected cost of a random ray, relative to the root's surface area
};

struct Bvh {
    std::vector<BvhNode> nodes; // nodes[0] is the root; empty if there were no triangles
    // triangleOrder[i] is the index in the input of the triangle the BVH stores
    // at position i. Apply it to the triangle arrays (see Reorder) before upload.
    std::vector<uint32_t> triangleOrder;
    BvhStats stats;
};

// Builds a binned-SAH BVH over the triangles. Does not modify the input.
Bvh BuildBvh(std::span<const TrianglePositions> triangles, const BvhBuildSettings& settings = {});

// Returns items permuted by a BVH's triangleOrder.
template <typename T>
std::vector<T> Reorder(std::span<const T> items, std::span<const uint32_t> order)
{
    std::vector<T> result;
    result.reserve(order.size());
    for (uint32_t index : order)
        result.push_back(items[index]);
    return result;
}

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction; // Need not be normalized; hit distances are in units of its length
};

struct Hit {
    float    t;        // Distance along the ray, in units of direction's length
    uint32_t triangle; // Index into the (reordered) triangle array
    float    u, v;     // Barycentrics of v1 and v2; v0's weight is 1 - u - v
};

// CPU reference implementations. The compute shader mirrors these, so they
// double as the ground truth for tests and GPU debugging. Both triangle faces
// are hit (no culling), and only hits with tMin < t < tMax count.
std::optional<Hit> IntersectTriangle(const Ray& ray, const TrianglePositions& triangle, float tMin, float tMax);
std::optional<Hit> IntersectBruteForce(const Ray& ray, std::span<const TrianglePositions> triangles, float tMin, float tMax);
// triangles must already be in BVH order (Reorder with bvh.triangleOrder).
std::optional<Hit> IntersectBvh(const Ray& ray, const Bvh& bvh, std::span<const TrianglePositions> triangles, float tMin, float tMax);

} // namespace Momo::RayTracing
