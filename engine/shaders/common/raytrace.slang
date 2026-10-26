// Path tracer: one thread per pixel follows a light path backwards from the
// camera, bouncing off matte surfaces, and averages the result over frames.
// Ray traversal lives in raytrace_traversal.slang, shared with traceonly.slang.

#include "raytrace_traversal.slang"

[vk::binding(4, 0)] [format("rgba16f")] RWTexture2D<float4> outputImage;

// Running average of every sample since the camera last moved
[vk::binding(6, 0)] [format("rgba32f")] RWTexture2D<float4> accumulationImage;

// Matches RayTracingPushConstants in VulkanRenderer.h
struct PushConstants {
    float4x4 inverseViewProjection; // NDC -> world
    float4   cameraPosition;        // w unused
    uint     frameIndex;            // Seeds the random numbers; changes every frame
    uint     sampleCount;           // Samples in the average including this one; 1 = start over
    uint     maxBounces;            // Bounces after the camera ray hits a surface; each adds
                                    // a closest-hit ray and a shadow ray per pixel (plus one
                                    // per point light)
    uint     pointLightCount;       // Entries of pointLights in use
};
[vk::push_constant] ConstantBuffer<PushConstants> pc;

// Matches RayTracing::RayTracingPointLight
struct PointLight {
    float4 position;  // w unused
    float4 intensity; // rgb: radiant intensity; a surface at distance d facing it receives intensity / d^2
};
[vk::binding(11, 0)] StructuredBuffer<PointLight> pointLights;

// The sun: direction toward it, and how much light it delivers to a surface
// facing it head-on (irradiance; the sky's brightness is the reference)
static const float3 lightDir = normalize(float3(0.5, 1.0, 0.3));
static const float3 SunIrradiance = float3(1.0, 0.95, 0.85) * 8.0;
static const float Pi = 3.14159265;
// Scales the light before tonemapping; higher is brighter
static const float Exposure = 1.0;
float3 Sky(float3 direction)
{
    float height = saturate(direction.y * 0.5 + 0.5);
    return lerp(float3(0.35, 0.35, 0.4), float3(0.55, 0.7, 0.95), height);
}

#ifdef CAPTURE_RAYS
static uint captureBounce; // Bounce of the path's current segment, for the queue choice

// Appends the ray to its queue; each queue holds at most one ray per pixel
void CaptureRay(Ray ray, bool shadow)
{
    if (captureBounce >= CaptureBounceLimit)
        return;
    uint2 size;
    outputImage.GetDimensions(size.x, size.y);
    uint capacity = size.x * size.y;
    uint queue = (shadow ? CaptureBounceLimit : 0) + captureBounce;
    uint slot;
    InterlockedAdd(captureCounts[queue], 1, slot);
    if (slot < capacity)
    {
        CapturedRay captured;
        captured.origin = float4(ray.origin, 0.0);
        captured.direction = float4(ray.direction, 0.0);
        capturedRays[queue * capacity + slot] = captured;
    }
}
#define CAPTURE_RAY(ray, shadow) CaptureRay(ray, shadow)
#define CAPTURE_BOUNCE(bounce) captureBounce = (bounce)
#else
#define CAPTURE_RAY(ray, shadow)
#define CAPTURE_BOUNCE(bounce)
#endif

// True if nothing blocks the way from origin toward the (infinitely far) sun
bool SeesSun(float3 origin)
{
    Ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = lightDir;
    CAPTURE_RAY(shadowRay, true);
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
    float3 emission;   // Light the surface gives off itself
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

    Material material = materials[triangle.materialIndex];
    surface.albedo = SampleBaseColor(material, InterpolateUV(triangle, hit)).rgb;
    surface.emission = material.emissive.rgb;
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

// Light from every point light reflected toward the viewer, one shadow ray
// each. Same matte reflection as the sun, but the light weakens with the
// square of the distance.
float3 DirectPointLights(Surface surface)
{
    float3 light = float3(0.0);
    for (uint i = 0; i < pc.pointLightCount; i++)
    {
        PointLight pointLight = pointLights[i];
        float3 toLight = pointLight.position.xyz - surface.position;
        float distanceSquared = dot(toLight, toLight);
        float distance = sqrt(distanceSquared);
        float3 direction = toLight / distance;
        float cosine = dot(surface.normal, direction);
        if (cosine <= 0.0 || dot(surface.faceNormal, direction) <= 0.0)
            continue;

        Ray shadowRay;
        shadowRay.origin = OffsetFromSurface(surface.position, surface.faceNormal);
        shadowRay.direction = direction;
        CAPTURE_RAY(shadowRay, true);
        Hit unused;
        // Stop just short of the light so nothing behind it counts as a blocker
        if (IntersectScene(shadowRay, 0.0, distance * 0.999, true, unused))
            continue;
        light += surface.albedo / Pi * pointLight.intensity.rgb * cosine / distanceSquared;
    }
    return light;
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
        CAPTURE_BOUNCE(bounce);
        CAPTURE_RAY(ray, false);
        Hit hit;
        if (!IntersectScene(ray, 0.0, Infinity, false, hit))
        {
            radiance += throughput * Sky(ray.direction);
            break;
        }

        Surface surface = GetSurface(ray, hit);
        radiance += throughput * (surface.emission + DirectSunlight(surface) + DirectPointLights(surface));
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
#ifdef COUNT_TRAVERSAL
    for (uint counter = 0; counter < CounterCount; counter++)
        threadCounts[counter] = 0;
#endif

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

#ifdef COUNT_TRAVERSAL
    // 64-bit add: carry into the high word when the low word wraps
    for (uint counter = 0; counter < CounterCount; counter++)
    {
        uint previous;
        InterlockedAdd(traversalCounters[counter * 2], threadCounts[counter], previous);
        if (previous + threadCounts[counter] < previous)
            InterlockedAdd(traversalCounters[counter * 2 + 1], 1u);
    }
#endif
}
