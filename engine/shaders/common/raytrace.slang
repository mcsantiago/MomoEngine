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

// Running average of every sample since the camera last moved
[vk::binding(6, 0)] [format("rgba32f")] RWTexture2D<float4> accumulationImage;

// Matches RayTracingPushConstants in VulkanRenderer.h
struct PushConstants {
    float4x4 inverseViewProjection; // NDC -> world
    float4   cameraPosition;        // w unused
    uint     frameIndex;            // Seeds the random numbers; changes every frame
    uint     sampleCount;           // Samples in the average including this one; 1 = start over
    uint     maxBounces;            // Bounces after the camera ray hits a surface; each adds
                                    // a closest-hit ray and a shadow ray per pixel
    uint     padding;
};
[vk::push_constant] ConstantBuffer<PushConstants> pc;

// The sun: direction toward it, and how much light it delivers to a surface
// facing it head-on (irradiance; the sky's brightness is the reference)
static const float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
static const float3 SunIrradiance = float3(1.0, 0.95, 0.85) * 8.0;
static const float Pi = 3.14159265;
// Scales the light before tonemapping; higher is brighter
static const float Exposure = 1.0;
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

// Finds the closest hit in (tMin, tMax). With anyHit, returns at the first
// hit found instead, which is all a shadow ray needs to know.
bool IntersectScene(Ray ray, float tMin, float tMax, bool anyHit, out Hit closest)
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
                    if (anyHit)
                        return true;
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

// True if nothing blocks the way from origin toward the (infinitely far) sun
bool SeesSun(float3 origin)
{
    Ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = lightDir;
    Hit unused;
    return !IntersectScene(shadowRay, 0.0, Infinity, true, unused);
}

// Floating-point error puts a computed hit point slightly above or below the
// surface, so a ray started exactly there can hit its own triangle ("shadow
// acne"). Start it a little off the surface instead, on the side given by
// normal. The offset grows with distance from the origin because float
// precision shrinks there.
float3 OffsetFromSurface(float3 position, float3 normal)
{
    float scale = max(1.0, max(abs(position.x), max(abs(position.y), abs(position.z))));
    return position + normal * (1e-4 * scale);
}

// What the path tracer needs to know about a hit point
struct Surface {
    float3 position;
    float3 normal;     // Smoothed (interpolated) normal, facing the incoming ray
    float3 faceNormal; // Flat triangle normal, facing the incoming ray
    float3 albedo;     // Fraction of light reflected, per channel (the base color)
};

Surface GetSurface(Ray ray, Hit hit)
{
    TriangleShading triangle = shading[hit.triangle];
    float w = 1.0 - hit.u - hit.v;

    Surface surface;
    surface.position = ray.origin + ray.direction * hit.t;
    surface.normal = normalize(triangle.normal0.xyz * w + triangle.normal1.xyz * hit.u + triangle.normal2.xyz * hit.v);
    // Triangles are hit from both sides; use the side the ray arrived from
    if (dot(surface.normal, ray.direction) > 0.0)
        surface.normal = -surface.normal;

    // Rays leave along the flat normal: the smoothed one can tilt into the
    // surface near silhouettes
    TrianglePositions positions = triangles[hit.triangle];
    surface.faceNormal = normalize(cross(positions.v1.xyz - positions.v0.xyz, positions.v2.xyz - positions.v0.xyz));
    if (dot(surface.faceNormal, ray.direction) > 0.0)
        surface.faceNormal = -surface.faceNormal;

    surface.albedo = SampleBaseColor(materials[triangle.materialIndex], InterpolateUV(triangle, hit)).rgb;
    return surface;
}

// Light from the sun reflected toward the viewer by a matte (Lambertian)
// surface: albedo / pi * irradiance * cos(angle to the sun), if the sun is visible
float3 DirectSunlight(Surface surface)
{
    float cosine = dot(surface.normal, lightDir);
    // A triangle whose flat side faces away from the sun shadows itself, even
    // if its smoothed normal still leans toward the sun. Otherwise trace a
    // shadow ray, but only if there's direct light to lose.
    if (cosine <= 0.0 || dot(surface.faceNormal, lightDir) <= 0.0 ||
        !SeesSun(OffsetFromSurface(surface.position, surface.faceNormal)))
        return float3(0.0);
    return surface.albedo / Pi * SunIrradiance * cosine;
}

// PCG hash (Jarzynski & Olano, "Hash Functions for GPU Rendering", 2020):
// cheap, and good enough that neighboring seeds give unrelated numbers
uint PcgHash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform in [0, 1). Uses the top 24 bits, which a float represents exactly
float NextRandom(inout uint rngState)
{
    rngState = PcgHash(rngState);
    return float(rngState >> 8) * (1.0 / 16777216.0);
}

// A random direction above the surface, more likely near the normal
// (probability proportional to cos(angle to the normal)). For a matte
// surface that cosine is exactly how much each direction's light counts, so
// sampling this way makes every bounce's weight simply the albedo.
float3 RandomCosineDirection(float3 normal, inout uint rngState)
{
    // Point on a unit disk, projected up onto the hemisphere
    float phi = 2.0 * Pi * NextRandom(rngState);
    float r2 = NextRandom(rngState);
    float r = sqrt(r2);
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - r2)));

    // Rotate from "z is up" to "normal is up" (orthonormal basis without a
    // branch on the normal's direction: Duff et al. 2017)
    float sign = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + normal.z);
    float b = normal.x * normal.y * a;
    float3 tangent = float3(1.0 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
    float3 bitangent = float3(b, sign + normal.y * normal.y * a, -normal.y);
    return normalize(local.x * tangent + local.y * bitangent + local.z * normal);
}

// Follows one light path backwards from the camera: at each surface, add the
// sunlight it reflects toward us, then continue in a random direction to find
// the light arriving from there. throughput is how much of that light makes
// it back to the camera after the surfaces passed so far.
float3 TracePath(Ray ray, inout uint rngState)
{
    float3 radiance = float3(0.0);
    float3 throughput = float3(1.0);

    for (uint bounce = 0; bounce <= pc.maxBounces; bounce++)
    {
        Hit hit;
        if (!IntersectScene(ray, 0.0, Infinity, false, hit))
        {
            radiance += throughput * Sky(ray.direction);
            break;
        }

        Surface surface = GetSurface(ray, hit);
        radiance += throughput * DirectSunlight(surface);
        if (bounce == pc.maxBounces)
            break;

        ray.direction = RandomCosineDirection(surface.normal, rngState);
        // Sampled around the smoothed normal, a direction can still point
        // into the actual triangle; that path ends here
        if (dot(ray.direction, surface.faceNormal) <= 0.0)
            break;
        ray.origin = OffsetFromSurface(surface.position, surface.faceNormal);
        throughput *= surface.albedo;
    }
    return radiance;
}

// Maps unbounded light values into displayable 0-1, rolling off highlights
// instead of clipping them (Narkowicz's fit of the ACES filmic curve)
float3 Tonemap(float3 color)
{
    color *= Exposure;
    return saturate((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compMain(uint3 threadId : SV_DispatchThreadID)
{
    uint2 size;
    outputImage.GetDimensions(size.x, size.y);
    if (threadId.x >= size.x || threadId.y >= size.y)
        return;

    // A different random sequence for every pixel and every frame
    uint rngState = PcgHash(threadId.x + PcgHash(threadId.y + PcgHash(pc.frameIndex)));

    // A random point inside the pixel rather than its center: averaged over
    // frames, edges come out smooth (antialiasing). Then NDC -> a world-space
    // point on the far plane. The projection already flips Y for Vulkan, so
    // row 0 maps to the top.
    float2 jitter = float2(NextRandom(rngState), NextRandom(rngState));
    float2 ndc = (float2(threadId.xy) + jitter) / float2(size) * 2.0 - 1.0;
    float4 target = mul(pc.inverseViewProjection, float4(ndc, 1.0, 1.0));

    Ray ray;
    ray.origin = pc.cameraPosition.xyz;
    ray.direction = normalize(target.xyz / target.w - ray.origin);

    float3 color = TracePath(ray, rngState);

    // Fold this sample into the running average: after n samples each one
    // has weight 1/n, without storing them all. The average stays in linear
    // light; only the displayed copy is tonemapped.
    float3 average = color;
    if (pc.sampleCount > 1)
    {
        float3 previous = accumulationImage[threadId.xy].rgb;
        average = previous + (color - previous) / float(pc.sampleCount);
    }
    accumulationImage[threadId.xy] = float4(average, 1.0);
    outputImage[threadId.xy] = float4(Tonemap(average), 1.0);
}
