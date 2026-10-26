#pragma once

#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>

namespace Momo::Renderer {
// The sun and sky every render path lights the scene with (lighting.slang).
// Point lights come from the scene file instead (VulkanRenderer::SetPointLights).
// Defaults: a low orange sunset sun under a sky that glows at the horizon.
struct SceneLighting
{
    // Where the sun is, in degrees: azimuth around the vertical axis from +X
    // toward +Z, and elevation above the horizon
    float sunAzimuth = 40.6f;
    float sunElevation = 20.8f;
    glm::vec3 sunColor{ 1.0f, 0.6f, 0.3f };
    float sunIntensity = 6.0f; // Irradiance on a surface facing the sun: sunColor * sunIntensity
    glm::vec3 skyGround{ 0.3f, 0.2f, 0.18f };  // Radiance below the horizon
    glm::vec3 skyHorizon{ 1.0f, 0.55f, 0.35f };
    glm::vec3 skyZenith{ 0.3f, 0.36f, 0.65f };
    float exposure = 0.6f; // Scales light before tonemapping; higher is brighter
    // Sun shadows of the raster modes, from a shadow map (the path tracer traces them)
    bool shadows = true;
    float shadowSoftness = 1.0f;   // Shadow map texels between filter taps; 0 = hardest edge
    float shadowNormalBias = 1.5f; // Shadow map texels to move a point along its normal before the lookup
    bool pointShadows = true;      // Cube shadow maps for the first few point lights
    float pointShadowBias = 1.5f;  // Cube map texels to move a point along its normal before the lookup
    // Screen-space ambient occlusion of the sky light (raster modes; the
    // path tracer's sky rays find occluders themselves)
    bool ambientOcclusion = true;
    float occlusionRadius = 1.0f;   // World units around a point searched for occluders
    float occlusionStrength = 1.5f; // Exponent on the unoccluded fraction; higher is darker
    bool showOcclusion = false;     // Debug: draw the occlusion alone

    // Unit vector toward the sun
    glm::vec3 SunDirection() const
    {
        float azimuth = glm::radians(sunAzimuth);
        float elevation = glm::radians(sunElevation);
        return glm::vec3(std::cos(azimuth) * std::cos(elevation), std::sin(elevation), std::sin(azimuth) * std::cos(elevation));
    }

    bool operator==(const SceneLighting&) const = default;
};

// How the raster pass turns light into color
enum class ShadingModel : uint32_t
{
    Lambert = 0, // Matte, like the path tracer's direct light
    Toon = 1,
};

// The toon shading model's look (RenderMode::Toon). Colors before
// tonemapping are in the same units as the light; outlineColor is final.
struct ToonSettings
{
    int bands = 3;              // Light levels from unlit to fully lit; 2 is classic cel shading
    float bandSoftness = 0.08f; // Blend between levels, as a fraction of a band
    glm::vec3 shadowTint{ 0.55f, 0.45f, 0.75f }; // Multiplies the sky light on the unlit side
    float specularStrength = 0.6f;
    float specularSize = 0.25f; // 0 = no highlight; 1 = the largest
    glm::vec3 rimColor{ 1.0f, 0.8f, 0.6f };
    float rimStrength = 0.35f;
    float rimWidth = 0.3f;      // How far the rim reaches in from the silhouette, 0-1
    glm::vec3 outlineColor{ 0.08f, 0.05f, 0.05f };
    float outlineWidth = 1.5f;  // Pixels; 0 = no outlines
    float depthThreshold = 0.08f; // Relative depth jump between neighbors that draws an outline
    float normalThreshold = 0.35f; // 1 - cos(angle) between neighbors' normals that draws an outline
};
} // namespace Momo::Renderer
