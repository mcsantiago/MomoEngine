struct VertexInput {
    [vk::location(0)] float3 position : POSITION;
    [vk::location(1)] float3 normal   : NORMAL;
    [vk::location(2)] float2 texCoord : TEXCOORD;
};

[vk::binding(0, 0)] Sampler2D baseColorTexture;
struct MaterialParams {
    float4 baseColorFactor;
    float  alphaCutoff;
    float3 _pad;
};
[vk::binding(1, 0)] ConstantBuffer<MaterialParams> material;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 worldNormal;
    float2 texCoord;
};

struct PushConstants {
    float4x4 projection;
    float4x4 view;
    float4x4 model;
};
[vk::push_constant] ConstantBuffer<PushConstants> pc;

static const float3 lightDir = normalize(float3(0.5, 1.0, 0.3));

[shader("vertex")]
VertexOutput vertMain(VertexInput input) {
    VertexOutput output;
    float4x4 mvp = mul(mul(pc.projection, pc.view), pc.model);
    output.sv_position = mul(mvp, float4(input.position, 1.0));
    output.worldNormal = mul((float3x3)pc.model, input.normal);
    output.texCoord = input.texCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert) : SV_Target
{
    float ndot1 = max(dot(normalize(inVert.worldNormal), lightDir), 0.0);
    float4 color = baseColorTexture.Sample(inVert.texCoord) * material.baseColorFactor;
    if (color.a < material.alphaCutoff) {
        discard;
    }
    return color;
}
