// What every pass drawing the scene's surfaces from the camera shares: the
// vertex stage and the material (descriptor set 0). The raster pass
// (triangle.slang) and its depth prepass (prepass.slang) both compile this
// vertex stage, so they compute bit-identical depths and the raster pass can
// test against the prepass's depth with LessOrEqual.

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
