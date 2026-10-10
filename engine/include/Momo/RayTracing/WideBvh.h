#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <glm/glm.hpp>
#include "Momo/RayTracing/Bvh.h"

namespace Momo::RayTracing {

// Compressed 8-wide BVH, after Ylitie, Karras & Laine, "Efficient Incoherent
// Ray Traversal on GPUs Through Compressed Wide BVHs" (HPG 2017).
//
// Up to 8 children per node makes the tree about a third as deep as a binary
// one, so a ray waits on about a third as many dependent node loads. Child
// boxes are stored as 8-bit offsets from the node's origin in units of a
// power of two per axis (rounded outwards, so a stored box always contains
// the true one), which fits a node in 80 bytes.

// Bit layout of CompressedWideNode::meta[slot]
inline constexpr uint32_t WideLeafMaxTriangles = 3;  // Triangle count is stored in unary in the top 3 bits
inline constexpr uint32_t WideInteriorMeta = 1u << 5; // Any non-zero value marks an interior child; internalMask says which

struct CompressedWideNode {
    glm::vec3 origin;          // Corner the quantized child boxes are offsets from
    uint8_t   exponent[3];     // Per axis: quantization step is 2^(exponent - 127), i.e. a float with these exponent bits
    uint8_t   internalMask;    // Bit s set: slot s holds an interior node
    uint32_t  childBase;       // Interior children are nodes childBase.. in slot order
    uint32_t  triangleBase;    // Leaf triangles are stored from here, in slot order
    // Per slot: 0 = empty. Leaf: low 5 bits = triangle offset from
    // triangleBase, top 3 bits = triangle count in unary (0b001, 0b011, 0b111).
    // Interior: non-zero (WideInteriorMeta).
    uint8_t   meta[8];
    uint8_t   lowX[8], lowY[8], lowZ[8];
    uint8_t   highX[8], highY[8], highZ[8];
};
static_assert(sizeof(CompressedWideNode) == 80, "CompressedWideNode must match the std430 layout in the shader");

struct WideBvhStats {
    BvhStats binary;        // The binary tree the wide one was collapsed from
    uint32_t nodeCount = 0;
    uint32_t maxDepth = 0;  // Wide nodes from the root to the deepest leaf
    double   buildMs = 0.0; // Wall-clock time of BuildWideBvh: binary build plus collapse
};

struct WideBvh {
    std::vector<CompressedWideNode> nodes; // nodes[0] is the root; empty if there were no triangles
    std::vector<uint32_t> triangleOrder;   // As Bvh::triangleOrder: input index of each stored triangle
    WideBvhStats stats;
};

// The settings the binary build uses before collapsing: leaves of at most
// WideLeafMaxTriangles, and room for the depth that needs
BvhBuildSettings WideBvhBinarySettings();

// The decoded box of a child slot, exactly as the shader computes it
void DecodeChildBounds(const CompressedWideNode& node, uint32_t slot, glm::vec3& boundsMin, glm::vec3& boundsMax);

// Ray octant: bit i set when direction component i is negative. Traversal
// visits slot (k ^ octant) k-th, which the builder arranges to be roughly
// near-to-far for rays of that octant.
uint32_t RayOctant(const glm::vec3& direction);

WideBvh BuildWideBvh(std::span<const TrianglePositions> triangles);

// CPU reference traversal, mirrored by the shader. triangles must already be
// in wide-BVH order (Reorder with WideBvh::triangleOrder).
std::optional<Hit> IntersectWideBvh(const Ray& ray, const WideBvh& bvh, std::span<const TrianglePositions> triangles, float tMin, float tMax);

} // namespace Momo::RayTracing
