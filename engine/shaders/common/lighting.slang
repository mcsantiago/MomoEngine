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
    float4x4 sunViewProjection;     // World -> the shadow map's clip space
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
    uint   shadowsEnabled;     // Raster only; the path tracer traces shadow rays
    float  shadowNormalOffset; // World units to move a point along its normal before the lookup
    float  shadowFilterRadius; // Shadow map texels between filter taps; larger = softer edges
    uint   pointShadowCount;   // Point lights 0..count-1 have a cube shadow map (raster only)
    float  pointShadowNear;    // Of every cube face's projection
    float  pointShadowFar;
    float  pointShadowBias;    // Cube map texels to move a point along its normal before the lookup
    ToonParams toon;
};
[vk::binding(0, 1)] ConstantBuffer<FrameUniforms> frame;

// Matches Renderer::GpuPointLight
struct PointLight {
    float4 position;  // w unused
    float4 intensity; // rgb: radiant intensity; a surface at distance d facing it receives intensity / d^2
};
[vk::binding(1, 1)] StructuredBuffer<PointLight> pointLights;

// Depth of the scene seen from the sun (shadow.slang), sampled with a depth
// comparison: 1 where the reference is no farther than the stored depth
[vk::binding(2, 1)] Sampler2DShadow shadowMap;

// Depth seen from point light i in each direction (shadow.slang, six faces
// per light), sampled with a depth comparison like shadowMap
static const uint MaxShadowedPointLights = 4;
[vk::binding(3, 1)] SamplerCubeShadow pointShadowMaps[MaxShadowedPointLights];

// Fraction of point light i reaching a point, like SunVisibility. The cube
// face that a direction falls on is the one of its largest component, and
// that component is the depth along the face's view axis, which the face's
// perspective projection maps to the stored depth.
float PointLightVisibility(uint index, float3 position, float3 normal)
{
    if (index >= frame.pointShadowCount)
        return 1.0;
    uint width, height;
    pointShadowMaps[index].GetDimensions(width, height);
    float3 fromLight = position - pointLights[index].position.xyz;
    float3 axes = abs(fromLight);
    float axisDistance = max(axes.x, max(axes.y, axes.z));
    // A 90 degree face spans 2 units at unit depth, so a texel covers 2 / width there
    fromLight += normal * (frame.pointShadowBias * axisDistance * 2.0 / float(width));
    axes = abs(fromLight);
    axisDistance = max(axes.x, max(axes.y, axes.z));
    float near = frame.pointShadowNear;
    float far = frame.pointShadowFar;
    // Beyond the map's reach, or inside its near plane: nothing recorded there
    if (axisDistance >= far || axisDistance <= near)
        return 1.0;
    float depth = far / (far - near) * (1.0 - near / axisDistance);
    return pointShadowMaps[index].SampleCmpLevelZero(fromLight, depth);
}

// Fraction of the sun reaching a point: 1 lit, 0 in shadow, in between at
// shadow edges. Averages 3x3 comparisons (percentage-closer filtering), each
// filtered bilinearly by the sampler. The point is first moved along its
// normal by about a texel, so a surface doesn't shadow itself from depth
// rounding ("shadow acne").
float SunVisibility(float3 position, float3 normal)
{
    if (frame.shadowsEnabled == 0)
        return 1.0;
    float4 clip = mul(frame.sunViewProjection, float4(position + normal * frame.shadowNormalOffset, 1.0));
    float3 ndc = clip.xyz / clip.w;
    float2 uv = ndc.xy * 0.5 + 0.5;
    // Outside the map: nothing there to cast a shadow
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z > 1.0)
        return 1.0;

    uint width, height;
    shadowMap.GetDimensions(width, height);
    float2 tapSpacing = frame.shadowFilterRadius / float2(width, height);
    float lit = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
            lit += shadowMap.SampleCmpLevelZero(uv + float2(x, y) * tapSpacing, ndc.z);
    return lit / 9.0;
}

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
