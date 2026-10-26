// Raster pass: draws the scene's meshes lit by the scene lighting
// (lighting.slang), with shadows from shadow maps (the sun's and up to
// MaxShadowedPointLights point lights') but no bounce light. frame.shadingModel picks
// how a surface turns light into color: Lambert (matte, like the path
// tracer's direct light) or toon. Also writes each pixel's normal for the
// outlines in composite.slang.

#include "lighting.slang"

struct VertexInput {
    [vk::location(0)] float3 position : POSITION;
    [vk::location(1)] float3 normal   : NORMAL;
    [vk::location(2)] float2 texCoord : TEXCOORD;
};

[vk::binding(0, 0)] Sampler2D baseColorTexture;
// Matches MaterialParams in VulkanMeshData.h
struct MaterialParams {
    float4 baseColorFactor;
    float4 emissive; // rgb: light the surface gives off
    float  alphaCutoff;
};
[vk::binding(1, 0)] ConstantBuffer<MaterialParams> material;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 worldPosition;
    float3 worldNormal;
    float2 texCoord;
};

struct FragmentOutput {
    float4 color  : SV_Target0;
    float4 normal : SV_Target1; // xyz: world-space normal; a = 1 where geometry was drawn
};

struct PushConstants {
    float4x4 projection;
    float4x4 view;
    float4x4 model;
};
[vk::push_constant] ConstantBuffer<PushConstants> pc;

// Normals transform by the inverse transpose, so they stay perpendicular to
// surfaces under non-uniform scale. The cofactor matrix is that up to a scale
// factor, which normalizing removes: its columns are cross products of the
// model's columns (where it sends each axis).
float3 TransformNormal(float3x3 model, float3 normal)
{
    float3 x = mul(model, float3(1.0, 0.0, 0.0));
    float3 y = mul(model, float3(0.0, 1.0, 0.0));
    float3 z = mul(model, float3(0.0, 0.0, 1.0));
    // A mirroring transform (negative determinant) would flip it inside out
    float handedness = dot(x, cross(y, z)) < 0.0 ? -1.0 : 1.0;
    return handedness * (normal.x * cross(y, z) + normal.y * cross(z, x) + normal.z * cross(x, y));
}

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;
    float4 worldPosition = mul(pc.model, float4(input.position, 1.0));
    output.sv_position = mul(mul(pc.projection, pc.view), worldPosition);
    output.worldPosition = worldPosition.xyz;
    output.worldNormal = TransformNormal((float3x3)pc.model, input.normal);
    output.texCoord = input.texCoord;
    return output;
}

// Matte reflection of the sun, the point lights and the sky
float3 ShadeLambert(float3 albedo, float3 normal, float3 position)
{
    float3 light = SkyAmbient(normal);
    float sunCosine = saturate(dot(normal, frame.sunDirection.xyz));
    if (sunCosine > 0.0)
        light += frame.sunIrradiance.rgb / Pi * sunCosine * SunVisibility(position, normal);
    for (uint i = 0; i < frame.pointLightCount; i++)
    {
        LightSample sample = SamplePointLight(pointLights[i], position);
        float cosine = saturate(dot(normal, sample.direction));
        if (cosine > 0.0)
            light += sample.irradiance / Pi * cosine * PointLightVisibility(i, position, normal);
    }
    return albedo * light;
}

// Quantizes how directly a surface faces a light (cosine) into toon.bands
// levels: 0 facing away, then evenly spaced steps up to 1. Each step blends
// over bandSoftness of a band instead of jumping.
float Band(float cosine)
{
    float steps = float(max(frame.toon.bands, 2u) - 1u);
    float scaled = saturate(cosine) * steps;
    float level = floor(scaled);
    float blend = smoothstep(0.0, max(frame.toon.bandSoftness, 1e-3), scaled - level);
    // A surface facing the light head-on is the top level, not the start of another
    return min(level + blend, steps) / steps;
}

// Banded lighting with a tinted shadow side, a hard specular highlight from
// the sun and a rim along silhouettes
float3 ShadeToon(float3 albedo, float3 normal, float3 position, float3 toViewer)
{
    ToonParams toon = frame.toon;
    float3 sunDirection = frame.sunDirection.xyz;
    float sunCosine = dot(normal, sunDirection);

    // In shadow counts as facing away: the bands then give shadows the same
    // crisp edges as the light falling off around a curve
    float sunVisibility = sunCosine > 0.0 ? SunVisibility(position, normal) : 0.0;
    float3 light = SkyAmbient(normal) * toon.shadowTint.rgb;
    light += frame.sunIrradiance.rgb / Pi * Band(sunCosine * sunVisibility);
    for (uint i = 0; i < frame.pointLightCount; i++)
    {
        LightSample sample = SamplePointLight(pointLights[i], position);
        float cosine = dot(normal, sample.direction);
        float visibility = cosine > 0.0 ? PointLightVisibility(i, position, normal) : 0.0;
        light += sample.irradiance / Pi * Band(cosine * visibility);
    }
    float3 color = albedo * light;

    // Blinn-Phong highlight cut to a solid spot: everywhere the half vector
    // is within the size's angle of the normal
    if (toon.specularSize > 0.0 && sunCosine > 0.0)
    {
        float3 halfVector = normalize(sunDirection + toViewer);
        float edge = 1.0 - toon.specularSize * 0.1;
        float highlight = smoothstep(edge - 0.002, edge + 0.002, dot(normal, halfVector)) * step(0.5, sunVisibility);
        color += toon.specularStrength * frame.sunIrradiance.rgb / Pi * highlight;
    }

    // Brightens surfaces seen edge-on
    float rim = 1.0 - saturate(dot(normal, toViewer));
    float rimStart = 1.0 - toon.rimWidth;
    color += toon.rimStrength * toon.rimColor.rgb * smoothstep(rimStart, rimStart + 0.05, rim);
    return color;
}

[shader("fragment")]
FragmentOutput fragMain(VertexOutput inVert, bool isFrontFace : SV_IsFrontFace)
{
    float4 baseColor = baseColorTexture.Sample(inVert.texCoord) * material.baseColorFactor;
    if (baseColor.a < material.alphaCutoff) {
        discard;
    }

    // Double-sided meshes show their back faces; light those from the side we see
    float3 normal = normalize(inVert.worldNormal);
    if (!isFrontFace)
        normal = -normal;
    float3 toViewer = normalize(frame.cameraPosition.xyz - inVert.worldPosition);

    float3 color = frame.shadingModel == ShadingToon
        ? ShadeToon(baseColor.rgb, normal, inVert.worldPosition, toViewer)
        : ShadeLambert(baseColor.rgb, normal, inVert.worldPosition);
    color += material.emissive.rgb;

    FragmentOutput output;
    output.color = float4(Tonemap(color), 1.0);
    output.normal = float4(normal, 1.0);
    return output;
}
