#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Momo/Assets/Handle.h"

namespace Momo {
// One placement of a model in the scene
struct SceneInstance
{
    Assets::ModelHandle model;
    glm::mat4 transform = glm::mat4(1.0f); // Model space to world space
    // Static instances make up the ray tracer's BVH and the collision
    // geometry, which are built once when the scene loads. Dynamic ones
    // (characters, things that move) can be moved every frame, but so far
    // only the raster modes draw them.
    bool isStatic = true;

    // Where the instance came from, for saving the scene
    std::string name;
    std::filesystem::path source; // The asset's glTF file
    // Instances from the scene file are saved with it. Ones game code adds
    // while running (a player character) aren't, or reloading would add them twice.
    bool saved = true;
};

class Scene
{
public:
    Scene() = default;
    ~Scene() = default;

    // Returns the instance's index, for SetTransform
    size_t AddInstance(SceneInstance instance) {
        m_Instances.push_back(std::move(instance));
        return m_Instances.size() - 1;
    }
    void SetTransform(size_t instance, const glm::mat4& transform) {
        m_Instances[instance].transform = transform;
    }
    const std::vector<SceneInstance>& GetInstances() const {
        return m_Instances;
    }
private:
    std::vector<SceneInstance> m_Instances;
};
} // namespace Momo
