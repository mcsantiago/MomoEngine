#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "Momo/Assets/Handle.h"

namespace Momo {
// Names an instance for as long as it's in the scene. Never reused, and
// unlike an index it survives other instances being removed.
using EntityId = uint64_t;

enum class InstanceKind
{
    // Part of the ray tracer's BVH and the collision geometry, which are
    // rebuilt in the background when a static instance changes
    Static,
    // Characters and things that move: can be moved every frame, but so far
    // only the raster modes draw them
    Dynamic,
    // A named place with no model, for game code to look up (a spawn point, say)
    Marker,
};

// One placement of a model in the scene, or a marker
struct SceneInstance
{
    EntityId id = 0; // Set by Scene::AddInstance
    Assets::ModelHandle model; // None for a marker
    glm::mat4 transform = glm::mat4(1.0f); // Model space to world space
    InstanceKind kind = InstanceKind::Static;

    // Where the instance came from, for saving the scene
    std::string name;
    std::filesystem::path source; // The asset's glTF file
    // Instances from the scene file are saved with it. Ones game code adds
    // while running (a player character) aren't, or reloading would add them twice.
    bool saved = true;

    bool IsStatic() const { return kind == InstanceKind::Static; }
    bool IsMarker() const { return kind == InstanceKind::Marker; }
};

class Scene
{
public:
    Scene() = default;
    ~Scene() = default;

    EntityId AddInstance(SceneInstance instance) {
        instance.id = m_NextId++;
        m_Instances.push_back(std::move(instance));
        return m_Instances.back().id;
    }
    // Null if the instance has been removed
    SceneInstance* Find(EntityId id) {
        auto it = std::find_if(m_Instances.begin(), m_Instances.end(), [id](const SceneInstance& instance) { return instance.id == id; });
        return it == m_Instances.end() ? nullptr : &*it;
    }
    const SceneInstance* Find(EntityId id) const {
        return const_cast<Scene*>(this)->Find(id);
    }
    void Remove(EntityId id) {
        std::erase_if(m_Instances, [id](const SceneInstance& instance) { return instance.id == id; });
    }
    const std::vector<SceneInstance>& GetInstances() const {
        return m_Instances;
    }
private:
    std::vector<SceneInstance> m_Instances;
    EntityId m_NextId = 1;
};
} // namespace Momo
