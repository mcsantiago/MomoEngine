// Screen-space ambient occlusion: how much of the sky each pixel of the
// depth prepass (prepass.slang) sees, judged from the depth buffer alone.
// Points scattered in the hemisphere above the surface are projected into the
// depth buffer; those behind what the camera sees there are inside geometry,
// so that direction is blocked. Compiled twice: this pass, and with
// SSAO_BLUR the depth-aware blur that removes its noise pattern. The raster
// pass (triangle.slang) reads the result through lighting.slang.

#include "lighting.slang"

[vk::binding(0, 0)] Sampler2D<float> depthTexture;
[vk::binding(1, 0)] [vk::image_format("r32f")] RWTexture2D<float> rawOcclusion;
[vk::binding(2, 0)] [vk::image_format("r32f")] RWTexture2D<float> occlusion;

static const uint SampleCount = 16;

// Turns the samples by a different angle at each pixel of a 4x4 block (a
// Bayer matrix, so neighbors differ as much as possible). The blur averages
// exactly one block, which cancels the pattern.
static const float RotationPattern[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };

float3 WorldPosition(int2 pixel, int2 size)
{
    pixel = clamp(pixel, int2(0), size - 1);
    float depth = depthTexture.Load(int3(pixel, 0));
    float2 ndc = (float2(pixel) + 0.5) / float2(size) * 2.0 - 1.0;
    float4 world = mul(frame.inverseViewProjection, float4(ndc, depth, 1.0));
    return world.xyz / world.w;
}

// The surface's normal from its neighbors' positions. On each axis the
// nearer neighbor is used, so a pixel on an edge takes its own surface's
// slope rather than one across the edge. Faceted, unlike the meshes' smooth
// normals, which keeps curved low-poly surfaces from occluding themselves.
float3 ReconstructNormal(int2 pixel, int2 size, float3 position)
{
    float3 right = WorldPosition(pixel + int2(1, 0), size) - position;
    float3 left = position - WorldPosition(pixel - int2(1, 0), size);
    float3 down = WorldPosition(pixel + int2(0, 1), size) - position;
    float3 up = position - WorldPosition(pixel - int2(0, 1), size);
    float3 dx = dot(right, right) < dot(left, left) ? right : left;
    float3 dy = dot(down, down) < dot(up, up) ? down : up;
    float3 normal = normalize(cross(dx, dy));
    return dot(normal, frame.cameraPosition.xyz - position) < 0.0 ? -normal : normal;
}

#ifndef SSAO_BLUR

[shader("compute")]
[numthreads(8, 8, 1)]
void compMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    depthTexture.GetDimensions(width, height);
    int2 size = int2(width, height);
    int2 pixel = int2(id.xy);
    if (any(pixel >= size))
        return;
    // Sky: nothing to occlude
    if (depthTexture.Load(int3(pixel, 0)) >= 1.0)
    {
        rawOcclusion[pixel] = 1.0;
        return;
    }

    float3 position = WorldPosition(pixel, size);
    float3 normal = ReconstructNormal(pixel, size, position);
    float3 tangent = normalize(cross(normal, abs(normal.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
    float3 bitangent = cross(normal, tangent);
    float pointDepth = mul(frame.viewProjection, float4(position, 1.0)).w;
    float rotation = (RotationPattern[(pixel.y & 3) * 4 + (pixel.x & 3)] + 0.5) / 16.0;

    float blocked = 0.0;
    for (uint i = 0; i < SampleCount; i++)
    {
        // Cosine-weighted directions on a golden-angle spiral, so blocked
        // samples weigh as much as the sky light they stand for
        float t = (float(i) + 0.5) / float(SampleCount);
        float angle = float(i) * 2.399963 + rotation * 2.0 * Pi;
        float r = sqrt(t);
        float3 direction = (tangent * cos(angle) + bitangent * sin(angle)) * r + normal * sqrt(1.0 - t);
        // More samples close to the point, where occluders shade it most
        float distance = frac(float(i) * 0.618034 + rotation);
        float3 samplePoint = position + direction * frame.occlusionRadius * lerp(0.1, 1.0, distance * distance);

        float4 clip = mul(frame.viewProjection, float4(samplePoint, 1.0));
        float2 uv = clip.xy / clip.w * 0.5 + 0.5;
        // Off screen: no depth there to judge by, so count it as open
        if (clip.w <= 0.0 || any(uv < 0.0) || any(uv >= 1.0))
            continue;
        float sceneDepth = LinearDepth(depthTexture.Load(int3(int2(uv * float2(size)), 0)));
        // Relative to depth, which loses precision with distance
        float bias = 0.01 + 0.002 * clip.w;
        if (sceneDepth < clip.w - bias)
            // An occluder far in front of the point (a separate object much
            // closer to the camera) doesn't shade it
            blocked += smoothstep(0.0, 1.0, frame.occlusionRadius / abs(pointDepth - sceneDepth));
    }
    rawOcclusion[pixel] = pow(1.0 - blocked / float(SampleCount), frame.occlusionStrength);
}

#else

// Averages the 4x4 block around each pixel, skipping pixels on other
// surfaces (much farther or nearer) so occlusion doesn't bleed across edges
[shader("compute")]
[numthreads(8, 8, 1)]
void compMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    depthTexture.GetDimensions(width, height);
    int2 size = int2(width, height);
    int2 pixel = int2(id.xy);
    if (any(pixel >= size))
        return;
    float depth = depthTexture.Load(int3(pixel, 0));
    if (depth >= 1.0)
    {
        occlusion[pixel] = 1.0;
        return;
    }

    float centerDepth = LinearDepth(depth);
    float sum = 0.0;
    float count = 0.0;
    for (int y = -2; y < 2; y++)
    {
        for (int x = -2; x < 2; x++)
        {
            int2 neighbor = clamp(pixel + int2(x, y), int2(0), size - 1);
            float neighborDepth = LinearDepth(depthTexture.Load(int3(neighbor, 0)));
            if (abs(neighborDepth - centerDepth) < 0.05 * centerDepth)
            {
                sum += rawOcclusion[neighbor];
                count += 1.0;
            }
        }
    }
    // The pixel itself always counts
    occlusion[pixel] = sum / count;
}

#endif
