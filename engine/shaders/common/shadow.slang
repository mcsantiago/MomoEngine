// Shadow pass: draws the scene's depth as seen from the sun into the shadow
// map that lighting.slang's SunVisibility() reads. Depth only; the fragment
// stage exists for alpha-masked materials (leaves), whose cut-out parts must
// not cast shadows.

struct VertexInput {
    [vk::location(0)] float3 position : POSITION;
    [vk::location(1)] float3 normal   : NORMAL;
    [vk::location(2)] float2 texCoord : TEXCOORD;
};

[vk::binding(0, 0)] Sampler2D baseColorTexture;
// Matches MaterialParams in VulkanMeshData.h
struct MaterialParams {
    float4 baseColorFactor;
    float4 emissive;
    float  alphaCutoff;
};
[vk::binding(1, 0)] ConstantBuffer<MaterialParams> material;

// Same layout as the raster pass's (PushConstantData): projection is the
// sun's view-projection, view is the identity
struct PushConstants {
    float4x4 projection;
    float4x4 view;
    float4x4 model;
};
[vk::push_constant] ConstantBuffer<PushConstants> pc;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float2 texCoord;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput input)
{
    VertexOutput output;
    output.sv_position = mul(mul(pc.projection, pc.view), mul(pc.model, float4(input.position, 1.0)));
    output.texCoord = input.texCoord;
    return output;
}

[shader("fragment")]
void fragMain(VertexOutput input)
{
    float alpha = baseColorTexture.Sample(input.texCoord).a * material.baseColorFactor.a;
    if (alpha < material.alphaCutoff)
        discard;
}
