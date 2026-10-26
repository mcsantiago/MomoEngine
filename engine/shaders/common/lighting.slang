// Scene lighting shared by every render path. The path tracer (raytrace.slang)
// and the raster passes (triangle.slang, composite.slang) read the same sun,
// sky and point lights from descriptor set 1; they differ only in how a
// surface turns that light into color.

static const float Pi = 3.14159265;

// Shading models of the raster pass (Renderer::ShadingModel)
static const uint ShadingLambert = 0;
static const uint ShadingToon = 1;

// Matches Renderer::ToonUniforms in VulkanRenderer.h (std140)
struct ToonParams {
    float4 shadowTint;     // rgb: multiplies the sky light on the unlit side
    float4 rimColor;       // rgb: light added along silhouettes, before exposure
    float4 outlineColor;   // rgb: displayed color (after tonemapping)
    uint   bands;          // Light levels from unlit to fully lit (2 = classic cel)
    float  bandSoftness;   // Width of the blend between levels, as a fraction of a band
    float  specularStrength;
    float  specularSize;   // 0 = none; larger = a bigger highlight
    float  rimStrength;
    float  rimWidth;       // 0-1: how far the rim reaches in from the silhouette
    float  outlineWidth;   // Pixels; 0 = no outlines
    float  depthThreshold; // Relative depth jump that counts as an edge
    float  normalThreshold;// 1 - cos(angle) between neighbors that counts as an edge
    float  padding0;
    float  padding1;
    float  padding2;
};

// Matches Renderer::FrameUniforms in VulkanRenderer.h (std140). Written once
// per frame.
struct FrameUniforms {
    float4x4 inverseViewProjection; // NDC -> world
    float4 cameraPosition;  // xyz
    float4 sunDirection;    // xyz: unit vector toward the sun
    float4 sunIrradiance;   // rgb: light delivered to a surface facing the sun head-on
    float4 skyGround;       // rgb: radiance of the sky below the horizon
    float4 skyHorizon;
    float4 skyZenith;
    float  exposure;        // Scales light before tonemapping; higher is brighter
    uint   pointLightCount; // Entries of pointLights in use
    uint   shadingModel;    // ShadingLambert or ShadingToon (raster only)
    float  nearPlane;       // Of the perspective projection, to linearize depth
    float  farPlane;
    float  padding0;
    float  padding1;
    float  padding2;
    ToonParams toon;
};
[vk::binding(0, 1)] ConstantBuffer<FrameUniforms> frame;

// Matches Renderer::GpuPointLight
struct PointLight {
    float4 position;  // w unused
    float4 intensity; // rgb: radiant intensity; a surface at distance d facing it receives intensity / d^2
};
[vk::binding(1, 1)] StructuredBuffer<PointLight> pointLights;

// Sunset-style gradient: the horizon color fading to the zenith color
// overhead, and to the ground color below
float3 Sky(float3 direction)
{
    if (direction.y < 0.0)
        return lerp(frame.skyHorizon.rgb, frame.skyGround.rgb, saturate(-direction.y * 4.0));
    return lerp(frame.skyHorizon.rgb, frame.skyZenith.rgb, saturate(direction.y * 2.0));
}

// Sky light reaching a surface (irradiance / pi), for passes that can't
// trace the sky. Ignores occlusion. Cosine-weighted averages of Sky() over a
// hemisphere: facing up sees mostly zenith (0.92 of the way from the
// horizon), facing down mostly ground; in between, a blend.
float3 SkyAmbient(float3 normal)
{
    float3 up = lerp(frame.skyHorizon.rgb, frame.skyZenith.rgb, 0.92);
    float3 down = lerp(frame.skyHorizon.rgb, frame.skyGround.rgb, 0.98);
    return lerp(down, up, normal.y * 0.5 + 0.5);
}

// Light arriving at a point from a point light
struct LightSample {
    float3 direction;  // Unit vector toward the light
    float  distance;
    float3 irradiance; // At normal incidence: intensity / d^2
};

LightSample SamplePointLight(PointLight light, float3 position)
{
    float3 toLight = light.position.xyz - position;
    float distanceSquared = max(dot(toLight, toLight), 1e-8);
    LightSample sample;
    sample.distance = sqrt(distanceSquared);
    sample.direction = toLight / sample.distance;
    sample.irradiance = light.intensity.rgb / distanceSquared;
    return sample;
}

// Maps unbounded light values into displayable 0-1, rolling off highlights
// instead of clipping them (Narkowicz's fit of the ACES filmic curve)
float3 Tonemap(float3 color)
{
    color *= frame.exposure;
    return saturate((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14));
}
