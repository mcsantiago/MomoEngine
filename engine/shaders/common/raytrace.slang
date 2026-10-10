// Primary-ray ray tracer: one thread per pixel builds a camera ray, finds the
// closest hit by traversing the scene BVH, and shades it. Traversal and the
// intersection tests mirror the CPU reference in engine/src/RayTracing/Bvh.cpp;
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
[vk::binding(4, 0)] [format("rgba16f")] RWTexture2D<float4> outputImage;
// Must match RayTracing::MaxRayTracingTextures. Unused slots hold a white texture.
static const uint MaxTextures = 256;
[vk::binding(5, 0)] Sampler2D baseColorTextures[MaxTextures];

struct PushConstants {
    float4x4 inverseView;
    float4x4 inverseProjection;
};
[vk::push_constant] ConstantBuffer<PushConstants> pc;

static const float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
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
    TriangleShading triangle = shading[triangleIndex];
    Material material = materials[triangle.materialIndex];
    return SampleBaseColor(material, InterpolateUV(triangle, hit)).a >= material.alphaCutoff;
}

bool IntersectScene(Ray ray, float tMin, float tMax, out Hit closest)
{
    closest = {};
    bool found = false;
    float3 inverseDirection = 1.0 / ray.direction;
    if (IntersectAabb(ray, inverseDirection, nodes[0], tMin, tMax) == Infinity)
        return false;

    uint stack[StackSize];
    uint stackSize = 0;
    uint nodeIndex = 0;

    while (true)
    {
        BvhNode node = nodes[nodeIndex];
        if (node.triangleCount > 0)
        {
            for (uint i = node.leftOrFirst; i < node.leftOrFirst + node.triangleCount; i++)
            {
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
                }
            }
        }
        else
        {
            uint nearChild = node.leftOrFirst;
            uint farChild = node.leftOrFirst + 1;
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

float3 Sky(float3 direction)
{
    float height = saturate(direction.y * 0.5 + 0.5);
    return lerp(float3(0.35, 0.35, 0.4), float3(0.55, 0.7, 0.95), height);
}

float3 Shade(Ray ray, Hit hit)
{
    TriangleShading triangle = shading[hit.triangle];
    float w = 1.0 - hit.u - hit.v;
    float3 normal = normalize(triangle.normal0.xyz * w + triangle.normal1.xyz * hit.u + triangle.normal2.xyz * hit.v);
    // Triangles are hit from both sides; light the side facing the camera
    if (dot(normal, ray.direction) > 0.0)
        normal = -normal;

    Material material = materials[triangle.materialIndex];
    float3 baseColor = SampleBaseColor(material, InterpolateUV(triangle, hit)).rgb;
    float diffuse = max(dot(normal, lightDir), 0.0);
    return baseColor * (0.15 + 0.85 * diffuse);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compMain(uint3 threadId : SV_DispatchThreadID)
{
    uint2 size;
    outputImage.GetDimensions(size.x, size.y);
    if (threadId.x >= size.x || threadId.y >= size.y)
        return;

    // Pixel center -> NDC -> a point on the far plane in view space. The
    // projection already flips Y for Vulkan, so row 0 maps to the top.
    float2 ndc = (float2(threadId.xy) + 0.5) / float2(size) * 2.0 - 1.0;
    float4 target = mul(pc.inverseProjection, float4(ndc, 1.0, 1.0));

    Ray ray;
    ray.origin = mul(pc.inverseView, float4(0.0, 0.0, 0.0, 1.0)).xyz;
    ray.direction = normalize(mul(pc.inverseView, float4(target.xyz / target.w, 0.0)).xyz);

    Hit hit;
    float3 color = IntersectScene(ray, 0.0, Infinity, hit) ? Shade(ray, hit) : Sky(ray.direction);
    outputImage[threadId.xy] = float4(color, 1.0);
}
