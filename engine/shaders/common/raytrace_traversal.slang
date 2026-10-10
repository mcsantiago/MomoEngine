// Ray traversal shared by the ray tracing shaders: the scene layouts and
// buffers, ray/triangle and ray/box tests, the alpha test and the BVH walk.
// The traversal mirrors the CPU reference in engine/src/RayTracing/Bvh.cpp;
// keep the two in sync. One deliberate difference: the shader also runs the
// alpha test for masked materials, which the CPU reference (no textures) skips.

// Layouts match engine/include/Momo/RayTracing/{Bvh,SceneGeometry}.h (std430)
// Compressed 8-wide BVH node (CompressedWideNode in WideBvh.h, Ylitie et al.
// 2017): 80 bytes, read as five uint4 rows so each is one 16-byte load.
//   row 0: origin.xyz (float bits), w = exponent x | y << 8 | z << 16 | internalMask << 24
//   row 1: childBase, triangleBase, meta[0..3], meta[4..7]
//   row 2: lowX[0..3], lowX[4..7], lowY[0..3], lowY[4..7]
//   row 3: lowZ[0..3], lowZ[4..7], highX[0..3], highX[4..7]
//   row 4: highY[0..3], highY[4..7], highZ[0..3], highZ[4..7]
static const uint WideNodeRows = 5;

struct TrianglePositions {
    float4 v0;    // w: 1 if the triangle's material is alpha masked
    float4 v1;
    float4 v2;
};

struct TriangleShading {
    float4 normal0;
    float4 normal1;
    float4 normal2;
    float2 uv0;
    float2 uv1;
    float2 uv2;
    uint   materialIndex;
    uint   padding;
};

struct Material {
    float4 baseColorFactor;
    float  alphaCutoff;      // -1 unless the material is alpha masked
    uint   baseColorTexture; // Index into baseColorTextures
    uint2  padding;
};

[vk::binding(0, 0)] StructuredBuffer<uint4> nodes; // WideNodeRows rows per node
[vk::binding(1, 0)] StructuredBuffer<TrianglePositions> triangles;
[vk::binding(2, 0)] StructuredBuffer<TriangleShading> shading;
[vk::binding(3, 0)] StructuredBuffer<Material> materials;
// Must match RayTracing::MaxRayTracingTextures. Unused slots hold a white texture.
static const uint MaxTextures = 256;
[vk::binding(5, 0)] Sampler2D baseColorTextures[MaxTextures];

// Traversal counters, only in the variant compiled with COUNT_TRAVERSAL (see
// engine/CMakeLists.txt), which benchmarks run separately from their timed
// frames so counting can't skew the timings. Each thread tallies in its own
// variables and adds them to the buffer once, as 64-bit counts (a frame can
// exceed 2^32 node reads). Must match VulkanRenderer::TraversalCounter.
#ifdef COUNT_TRAVERSAL
[vk::binding(7, 0)] RWStructuredBuffer<uint> traversalCounters; // (low, high) pairs
static const uint CounterClosestRays = 0;
static const uint CounterShadowRays = 1;
static const uint CounterSteps = 2;          // Traversal loop iterations
static const uint CounterNodeReads = 3;      // BVH nodes read from memory
static const uint CounterBoxTests = 4;
static const uint CounterTriangleTests = 5;
static const uint CounterAlphaTests = 6;
static const uint CounterCount = 7;
static uint threadCounts[CounterCount];
#define COUNT(counter, amount) threadCounts[counter] += (amount)
#else
#define COUNT(counter, amount)
#endif

// Ray capture for the trace-only experiment: the path tracer's CAPTURE_RAYS
// variant records every ray it traces in one frame, one queue per (kind,
// bounce), and traceonly.slang traces them again in a kernel that does
// nothing else. Must match VulkanRenderer's capture buffers.
#if defined(CAPTURE_RAYS) || defined(TRACE_ONLY)
struct CapturedRay {
    float4 origin;    // w unused
    float4 direction; // w unused
};
static const uint CaptureBounceLimit = 6;                   // Camera ray + 5 bounces
static const uint CaptureQueueCount = 2 * CaptureBounceLimit; // Closest-hit queues, then shadow queues
[vk::binding(8, 0)] RWStructuredBuffer<CapturedRay> capturedRays; // Queue q: [q * capacity, (q + 1) * capacity)
[vk::binding(9, 0)] RWStructuredBuffer<uint> captureCounts;       // Rays appended to each queue
#endif

static const float Infinity = asfloat(0x7f800000);
// Traversal pushes at most one entry per level, and the builder keeps the
// wide tree shallower than this (WideTraversalStackSize in WideBvh.cpp)
static const uint StackSize = 32;

struct Ray {
    float3 origin;
    float3 direction;
};

struct Hit {
    float t;
    uint  triangle;
    float u, v;
};

// Möller-Trumbore. Conditions are written as !(in range) so NaNs from
// degenerate triangles or parallel rays are rejected rather than accepted.
bool IntersectTriangle(Ray ray, TrianglePositions triangle, float tMin, float tMax, out Hit hit)
{
    hit = {};
    float3 v0 = triangle.v0.xyz;
    float3 edge1 = triangle.v1.xyz - v0;
    float3 edge2 = triangle.v2.xyz - v0;

    float3 p = cross(ray.direction, edge2);
    float determinant = dot(edge1, p);
    if (determinant == 0.0)
        return false;
    float inverseDeterminant = 1.0 / determinant;

    float3 s = ray.origin - v0;
    float u = dot(s, p) * inverseDeterminant;
    if (!(u >= 0.0 && u <= 1.0))
        return false;

    float3 q = cross(s, edge1);
    float v = dot(ray.direction, q) * inverseDeterminant;
    if (!(v >= 0.0 && u + v <= 1.0))
        return false;

    float t = dot(edge2, q) * inverseDeterminant;
    if (!(t > tMin && t < tMax))
        return false;

    hit.t = t;
    hit.u = u;
    hit.v = v;
    return true;
}

// Returns the entry distance into the box, or infinity on a miss. A zero
// direction component gives an infinite inverse, and 0 * inf = NaN when the
// origin lies on that slab's plane; NaN counts as unbounded so rays running
// along a box face still hit it.
float IntersectAabb(Ray ray, float3 inverseDirection, float3 boundsMin, float3 boundsMax, float tMin, float tMax)
{
    float3 t0 = (boundsMin - ray.origin) * inverseDirection;
    float3 t1 = (boundsMax - ray.origin) * inverseDirection;
    float3 tNear = min(select(isnan(t0), -Infinity, t0), select(isnan(t1), -Infinity, t1));
    float3 tFar = max(select(isnan(t0), Infinity, t0), select(isnan(t1), Infinity, t1));
    float tEnter = max(max(tNear.x, tNear.y), max(tNear.z, tMin));
    float tExit = min(min(tFar.x, tFar.y), min(tFar.z, tMax));
    return tEnter <= tExit ? tEnter : Infinity;
}

float2 InterpolateUV(TriangleShading triangle, Hit hit)
{
    return triangle.uv0 * (1.0 - hit.u - hit.v) + triangle.uv1 * hit.u + triangle.uv2 * hit.v;
}

// Neighboring pixels can hit different materials, so the texture index must
// be marked non-uniform. Compute shaders have no derivatives to pick a mip
// level from, and the textures have no mips anyway, so sample level 0.
float4 SampleBaseColor(Material material, float2 uv)
{
    return baseColorTextures[NonUniformResourceIndex(material.baseColorTexture)].SampleLevel(uv, 0.0) * material.baseColorFactor;
}

// The ray tracing equivalent of the raster shader's discard: a hit on a
// transparent part of a masked triangle (e.g. around a leaf) doesn't count
bool PassesAlphaTest(uint triangleIndex, Hit hit)
{
    COUNT(CounterAlphaTests, 1);
    TriangleShading triangle = shading[triangleIndex];
    Material material = materials[triangle.materialIndex];
    return SampleBaseColor(material, InterpolateUV(triangle, hit)).a >= material.alphaCutoff;
}

// Byte `index` (0-7) of the 8 bytes stored in two uints
uint ByteOf(uint low4, uint high4, uint index)
{
    return ((index < 4 ? low4 : high4) >> ((index & 3) * 8)) & 0xFF;
}

// Tests the triangles of the leaf in `slot` of a node (meta, base from row 1),
// shortening tMax to each closer hit. With anyHit, stops at the first.
bool IntersectWideLeaf(Ray ray, uint4 row1, uint slot, float tMin, inout float tMax, bool anyHit, inout Hit closest)
{
    uint meta = ByteOf(row1.z, row1.w, slot);
    uint first = row1.y + (meta & 31);
    uint count = countbits(meta >> 5); // Unary-coded triangle count
    bool found = false;
    for (uint i = first; i < first + count; i++)
    {
        COUNT(CounterTriangleTests, 1);
        Hit hit;
        TrianglePositions triangle = triangles[i];
        // Only masked triangles pay for the texture lookup
        if (IntersectTriangle(ray, triangle, tMin, tMax, hit) &&
            (triangle.v0.w == 0.0 || PassesAlphaTest(i, hit)))
        {
            closest = hit;
            closest.triangle = i;
            tMax = hit.t;
            found = true;
            if (anyHit)
                return true;
        }
    }
    return found;
}

// Finds the closest hit in (tMin, tMax). With anyHit, returns at the first
// hit found instead, which is all a shadow ray needs to know.
//
// Mirrors IntersectWideBvh in WideBvh.cpp. Visiting a node tests all its
// children's boxes at once into an 8-bit hit mask, ordered so the lowest bit
// is (roughly) the nearest child: bit k is slot k ^ octant, and the builder
// puts in slot s the child that rays of octant s reach first. Leaves are
// tested in place; descending into an interior child pushes the rest of the
// mask as one stack entry, so the stack holds at most one entry per level.
bool IntersectScene(Ray ray, float tMin, float tMax, bool anyHit, out Hit closest)
{
    closest = {};
    bool found = false;
    float3 inverseDirection = 1.0 / ray.direction;
    uint octant = (ray.direction.x < 0.0 ? 1u : 0u) | (ray.direction.y < 0.0 ? 2u : 0u) | (ray.direction.z < 0.0 ? 4u : 0u);
    COUNT(anyHit ? CounterShadowRays : CounterClosestRays, 1);

    // Entries: node index << 8 | its children still to visit (mask bits)
    uint stack[StackSize];
    uint stackSize = 0;
    uint nodeIndex = 0;

    while (true)
    {
        // Visit a node: decode and test all eight child boxes
        uint4 row0 = nodes[nodeIndex * WideNodeRows + 0];
        uint4 row1 = nodes[nodeIndex * WideNodeRows + 1];
        uint4 row2 = nodes[nodeIndex * WideNodeRows + 2];
        uint4 row3 = nodes[nodeIndex * WideNodeRows + 3];
        uint4 row4 = nodes[nodeIndex * WideNodeRows + 4];
        COUNT(CounterSteps, 1);
        COUNT(CounterNodeReads, 1);

        float3 origin = asfloat(row0.xyz);
        // Quantization step per axis: a float with the stored exponent bits
        float3 step = float3(asfloat((row0.w & 0xFF) << 23), asfloat(((row0.w >> 8) & 0xFF) << 23), asfloat(((row0.w >> 16) & 0xFF) << 23));
        uint hits = 0;
        for (uint k = 0; k < 8; k++)
        {
            uint slot = k ^ octant;
            if (ByteOf(row1.z, row1.w, slot) == 0)
                continue;
            float3 low = origin + float3(ByteOf(row2.x, row2.y, slot), ByteOf(row2.z, row2.w, slot), ByteOf(row3.x, row3.y, slot)) * step;
            float3 high = origin + float3(ByteOf(row3.z, row3.w, slot), ByteOf(row4.x, row4.y, slot), ByteOf(row4.z, row4.w, slot)) * step;
            COUNT(CounterBoxTests, 1);
            if (IntersectAabb(ray, inverseDirection, low, high, tMin, tMax) != Infinity)
                hits |= 1u << k;
        }

        // Work through groups, testing leaves in place, until an interior
        // child to descend into turns up (or nothing is left)
        uint groupNode = nodeIndex;
        uint internalMask = row0.w >> 24;
        bool descend = false;
        while (true)
        {
            while (hits != 0)
            {
                uint k = firstbitlow(hits);
                hits &= hits - 1;
                uint slot = k ^ octant;
                if ((internalMask >> slot) & 1)
                {
                    if (hits != 0 && stackSize < StackSize)
                        stack[stackSize++] = (groupNode << 8) | hits;
                    nodeIndex = row1.x + countbits(internalMask & ((1u << slot) - 1));
                    descend = true;
                    break;
                }
                if (IntersectWideLeaf(ray, row1, slot, tMin, tMax, anyHit, closest))
                {
                    found = true;
                    if (anyHit)
                        return true;
                }
            }
            if (descend || stackSize == 0)
                break;
            // Resume a pushed group: its node's header and child references
            stackSize--;
            groupNode = stack[stackSize] >> 8;
            hits = stack[stackSize] & 0xFF;
            row0 = nodes[groupNode * WideNodeRows + 0];
            row1 = nodes[groupNode * WideNodeRows + 1];
            internalMask = row0.w >> 24;
        }
        if (!descend)
            return found;
    }
}
