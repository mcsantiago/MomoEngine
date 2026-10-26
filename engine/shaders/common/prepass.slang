// Depth prepass: the scene's depth from the camera, before the raster pass,
// so ssao.slang can compute ambient occlusion that the raster pass then reads.
// Alpha-masked cut-outs are discarded as in the raster pass.

#include "surface.slang"

[shader("fragment")]
void fragMain(VertexOutput input)
{
    float alpha = baseColorTexture.Sample(input.texCoord).a * material.baseColorFactor.a;
    if (alpha < material.alphaCutoff)
        discard;
}
