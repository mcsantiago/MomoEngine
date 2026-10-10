// Ray traversal shared by the ray tracing shaders: the scene layouts and
// buffers, ray/triangle and ray/box tests, the alpha test and the BVH walk.
// The traversal mirrors the CPU reference in engine/src/RayTracing/Bvh.cpp;
// keep the two in sync. One deliberate difference: the shader also runs the
// alpha test for masked materials, which the CPU reference (no textures) skips.

// Layouts match engine/include/Momo/RayTracing/{Bvh,SceneGeometry}.h (std430)
struct BvhNode {
    float3 boundsMin;
    uint   leftOrFirst;   // Interior: left child (right = left + 1). Leaf: first triangle
    float3 boundsMax;
    uint   triangleCount; // 0 for interior nodes
};

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

[vk::binding(0, 0)] StructuredBuffer<BvhNode> nodes;
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
static const uint CounterNodeReads = 3;      // BvhNode entries read from memory
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
// Near-first traversal pushes at most one node per level, so the BVH build's
// maxDepth (32 by default) bounds this
static const uint StackSize = 64;

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
float IntersectAabb(Ray ray, float3 inverseDirection, BvhNode node, float tMin, float tMax)
{
    float3 t0 = (node.boundsMin - ray.origin) * inverseDirection;
    float3 t1 = (node.boundsMax - ray.origin) * inverseDirection;
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

// Finds the closest hit in (tMin, tMax). With anyHit, returns at the first
// hit found instead, which is all a shadow ray needs to know.
bool IntersectScene(Ray ray, float tMin, float tMax, bool anyHit, out Hit closest)
{
    closest = {};
    bool found = false;
    float3 inverseDirection = 1.0 / ray.direction;
    COUNT(anyHit ? CounterShadowRays : CounterClosestRays, 1);
    COUNT(CounterNodeReads, 1);
    COUNT(CounterBoxTests, 1);
    if (IntersectAabb(ray, inverseDirection, nodes[0], tMin, tMax) == Infinity)
        return false;

    uint stack[StackSize];
    uint stackSize = 0;
    uint nodeIndex = 0;

    while (true)
    {
        BvhNode node = nodes[nodeIndex];
        COUNT(CounterSteps, 1);
        COUNT(CounterNodeReads, 1);
        if (node.triangleCount > 0)
        {
            for (uint i = node.leftOrFirst; i < node.leftOrFirst + node.triangleCount; i++)
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
        }
        else
        {
            uint nearChild = node.leftOrFirst;
            uint farChild = node.leftOrFirst + 1;
            COUNT(CounterNodeReads, 2);
            COUNT(CounterBoxTests, 2);
            float nearDistance = IntersectAabb(ray, inverseDirection, nodes[nearChild], tMin, tMax);
            float farDistance = IntersectAabb(ray, inverseDirection, nodes[farChild], tMin, tMax);
            if (farDistance < nearDistance)
            {
                uint swapChild = nearChild; nearChild = farChild; farChild = swapChild;
                float swapDistance = nearDistance; nearDistance = farDistance; farDistance = swapDistance;
            }

            if (nearDistance != Infinity)
            {
                if (farDistance != Infinity && stackSize < StackSize)
                    stack[stackSize++] = farChild;
                nodeIndex = nearChild;
                continue;
            }
        }

        if (stackSize == 0)
            break;
        nodeIndex = stack[--stackSize];
    }
    return found;
}
