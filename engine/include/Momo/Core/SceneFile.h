#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Momo/Assets/ModelSource.h"

// Scene files: JSON listing the scene's entities (each a glTF asset placed in
// the world) plus an optional camera and lighting. For example:
//
//   {
//     "version": 1,
//     "camera": { "position": [0, 1.6, 5], "target": [0, 1, 0] },
//     "lighting": {
//       "sun": { "towardSun": [0.5, 0.8, 0.3], "color": [1, 0.93, 0.82], "intensity": 5 },
//       "sky": { "zenith": [0.2, 0.4, 0.85], "horizon": [0.6, 0.7, 0.85], "ground": [0.3, 0.3, 0.25], "exposure": 0.55 }
//     },
//     "entities": [
//       { "name": "Street", "asset": "Market/Market.glb" },
//       { "name": "Crate", "asset": "../Models/Crate.glb", "position": [2, 0, 4],
//         "rotation": [0, 35, 0], "scale": [1, 1, 1], "static": true }
//     ]
//   }
//
// Asset paths are relative to the scene file. Rotations are Euler angles in
// degrees, applied about X, then Y, then Z (world axes; Blender's default
// "XYZ Euler"). Everything but "entities" and each entity's "asset" is
// optional: entities default to the origin, unrotated, at scale 1, and
// static. Without a camera or lighting, the scene uses the first ones its
// assets bring (a glTF camera, a directional light, sky extras).
namespace Momo
{
    inline constexpr int SceneFileVersion = 1;

    struct EntityDescription
    {
        std::string name;
        std::filesystem::path asset; // As written in the file: relative to it, or absolute
        glm::vec3 position{ 0.0f };
        glm::vec3 rotation{ 0.0f }; // Euler degrees, X then Y then Z
        glm::vec3 scale{ 1.0f };
        // Static entities make up the ray tracer's BVH and the collision
        // geometry; dynamic ones can move but only the raster modes draw them
        bool isStatic = true;

        glm::mat4 Transform() const;
    };

    struct SceneCameraDescription
    {
        glm::vec3 position;
        glm::vec3 target;
    };

    struct SceneDescription
    {
        std::optional<SceneCameraDescription> camera;
        std::optional<Assets::SunSource> sun;   // towardSun in world space
        Assets::SkySource sky;                  // Unset fields come from the assets, then the defaults
        std::vector<EntityDescription> entities;
    };

    // Parses a scene from JSON text; on failure returns nothing and sets error
    std::optional<SceneDescription> ParseScene(const std::string& json, std::string& error);
    std::optional<SceneDescription> ReadSceneFile(const std::filesystem::path& path, std::string& error);
    std::string SerializeScene(const SceneDescription& scene);
    bool WriteSceneFile(const SceneDescription& scene, const std::filesystem::path& path, std::string& error);

    // Translation, then rotation (Euler degrees, X then Y then Z), then scale
    glm::mat4 ComposeTransform(glm::vec3 position, glm::vec3 rotationDegrees, glm::vec3 scale);
    // The inverse of ComposeTransform, for transforms without shear. A
    // mirroring transform comes back as a rotation with all scales positive,
    // which is not the same transform.
    void DecomposeTransform(const glm::mat4& transform, glm::vec3& position, glm::vec3& rotationDegrees, glm::vec3& scale);
}
