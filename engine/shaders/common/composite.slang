// Composite pass, drawn over the raster pass's image as one full-screen
// triangle: fills the pixels nothing was drawn on with the sky, and in toon
// mode draws outlines where depth or normals jump between neighboring pixels.

#include "lighting.slang"

[vk::binding(0, 0)] Sampler2D<float> depthTexture;
[vk::binding(1, 0)] Sampler2D normalTexture; // xyz: world-space normal; a = 0 where nothing was drawn

struct VertexOutput {
    float4 sv_position : SV_Position;
    float2 ndc;
};

// Vertex 0, 1, 2 -> a triangle covering the whole screen
[shader("vertex")]
VertexOutput vertMain(uint vertexId : SV_VertexID)
{
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    VertexOutput output;
    output.ndc = uv * 2.0 - 1.0;
    output.sv_position = float4(output.ndc, 0.0, 1.0);
    return output;
}

// 1 if the pixel lies on an outline. Silhouettes: a neighbor is farther away
// by more than depthThreshold (relative, so it holds at any distance); the
// outline goes on the nearer pixel only, so it stays on the object. Creases:
// a neighbor's normal turns by more than normalThreshold; checked toward +x
// and +y only, so a crease is one line rather than one on each side.
float Outline(int2 pixel, int2 size)
{
    ToonParams toon = frame.toon;
    int width = max(1, int(round(toon.outlineWidth)));
    float depth = LinearDepth(depthTexture.Load(int3(pixel, 0)));
    float4 normal = normalTexture.Load(int3(pixel, 0));
    // Nothing drawn here: silhouettes land on the object's side instead
    if (normal.a == 0.0)
        return 0.0;

    const int2 offsets[4] = { int2(1, 0), int2(0, 1), int2(-1, 0), int2(0, -1) };
    for (int i = 0; i < 4; i++)
    {
        int2 neighbor = clamp(pixel + offsets[i] * width, int2(0), size - 1);
        float neighborDepth = LinearDepth(depthTexture.Load(int3(neighbor, 0)));
        if ((neighborDepth - depth) / depth > toon.depthThreshold * width)
            return 1.0;
        float4 neighborNormal = normalTexture.Load(int3(neighbor, 0));
        if (i < 2 && neighborNormal.a > 0.0 && 1.0 - dot(normal.xyz, neighborNormal.xyz) > toon.normalThreshold)
            return 1.0;
    }
    return 0.0;
}

[shader("fragment")]
float4 fragMain(VertexOutput input) : SV_Target
{
    int2 pixel = int2(input.sv_position.xy);
    uint width, height;
    depthTexture.GetDimensions(width, height);

    if (frame.shadingModel == ShadingToon && frame.toon.outlineWidth > 0.0)
    {
        float outline = Outline(pixel, int2(width, height));
        if (outline > 0.0)
            // Outlines thinner than a pixel fade instead
            return float4(frame.toon.outlineColor.rgb, saturate(frame.toon.outlineWidth));
    }

    if (normalTexture.Load(int3(pixel, 0)).a > 0.0)
        discard;
    // The far plane point under this pixel gives the view direction
    float4 target = mul(frame.inverseViewProjection, float4(input.ndc, 1.0, 1.0));
    float3 direction = normalize(target.xyz / target.w - frame.cameraPosition.xyz);
    return float4(Tonemap(Sky(direction)), 1.0);
}
