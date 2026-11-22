#pragma once
#include <array>
#include <chrono>
#include <unordered_map>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>
#include <utility>
#include <memory>
#include <future>
#include "Momo/Window.h"
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Renderer/Camera.h"
#include "Momo/Assets/ModelLoader.h"
#include "Momo/Assets/AssetRegistry.h"
#include "Momo/Cache/DataCache.h"
#include "Momo/Core/Scene.h"
#include "Momo/Core/SceneFile.h"
#include "Momo/Core/Benchmark.h"
#include "Momo/RayTracing/WideBvh.h"


namespace Momo
{
    // Renders one path traced frame of the scene to a file, then exits
    struct ScreenshotSpecification
    {
        std::filesystem::path OutputPath; // Written as JPEG
        uint32_t Samples = 256;           // Path traced samples averaged before saving
    };

    struct ApplicationSpecification
    {
        std::string Name = "Untitled";
        WindowSpecification WindowSpec{};
        Renderer::RenderMode InitialRenderMode = Renderer::RenderMode::Raster; // Cycled at runtime with P
        // Scene to load, relative to the binary dir: a scene file (.json, see
        // SceneFile.h) or a single glTF file. Its camera, if it has one, is
        // where the view starts
        std::filesystem::path ScenePath = "Assets/Models/Sponza/Sponza.gltf";
        // Set to run the ray tracing benchmark instead of interactive mode; the app exits when done
        std::optional<BenchmarkSpecification> Benchmark;
        // Set to save a screenshot from the scene's camera instead of running interactively
        std::optional<ScreenshotSpecification> Screenshot;
    };

    class Application;

    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual void OnAttach() {}
        virtual void OnDetach() {}
        // Called every frame before the camera's own update and the draw
        virtual void OnUpdate(float dt) {}
        // Called every frame (except in benchmarks) inside the Dear ImGui
        // frame, to draw the layer's UI
        virtual void OnImGui() {}
        // Game layers stop updating while the application is in editing mode;
        // an editor's own layer keeps going
        virtual bool UpdatesWhileEditing() const { return false; }
    protected:
        // The application the layer was pushed onto (set before OnAttach)
        Application& App() { return *m_App; }
        const Application& App() const { return *m_App; }
    private:
        friend class Application;
        Application* m_App = nullptr;
    };

    // A hit from Application::Raycast
    struct RaycastHit
    {
        float distance;   // Along the ray, in world units
        glm::vec3 point;
        glm::vec3 normal; // Unit length, facing back along the ray
    };

    // A hit from Application::PickInstance
    struct PickHit
    {
        EntityId entity;
        float distance; // Along the ray, in units of the direction's length
        glm::vec3 point;
    };

    // An axis-aligned box; empty (min > max) for a model with no vertices
    struct Bounds
    {
        glm::vec3 min;
        glm::vec3 max;
    };

    class Application {
    public:
        explicit Application(const ApplicationSpecification& spec);
        ~Application();

        template<typename TLayer, typename... Args>
        TLayer* PushLayer(Args&&... args)
        {
            auto layer = std::make_unique<TLayer>(std::forward<Args>(args)...);
            TLayer* raw = layer.get();
            raw->m_App = this;
            m_Layers.emplace_back(std::move(layer));
            raw->OnAttach();
            return raw;
        }

        // --- For layers ---------------------------------------------------
        const Input::InputState& GetInput() const { return m_InputState; }
        Camera& GetCamera() { return *m_Camera; }
        const Camera& GetCamera() const { return *m_Camera; }
        // With the free camera off, the camera stays wherever a layer puts it
        // instead of flying around with WASD and the mouse
        void SetFreeCamera(bool enabled) { m_FreeCamera = enabled; }
        // Loads a glTF model into the asset registry without placing it (or
        // returns the one already loaded from that file)
        std::optional<Assets::ModelHandle> LoadModel(const std::filesystem::path& path);
        // Places a model in the active scene for game code to move around
        // (a character). It isn't saved with the scene, and it stays when
        // another scene is opened.
        EntityId AddDynamicInstance(Assets::ModelHandle model, const glm::mat4& transform);
        // Does nothing if the instance has been removed
        void SetInstanceTransform(EntityId instance, const glm::mat4& transform);
        // The first hit along a ray (direction of unit length) against the
        // static scene, within maxDistance. The collision geometry is built
        // with the ray tracer's BVH on a worker thread, so there are no hits
        // until IsCollisionReady(). Alpha masked surfaces (leaves) count as solid.
        std::optional<RaycastHit> Raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance) const;
        bool IsCollisionReady() const { return m_Collision.has_value(); }
        // Writes the scene as a scene file: its entities where they are now,
        // the scene camera and the current sun and sky
        bool SaveScene(const std::filesystem::path& path, std::string& error);

        // --- For editors --------------------------------------------------
        // Changing anything static rebuilds the ray tracing and collision
        // geometry in the background, a moment after the last change. Until
        // that finishes the ray traced view and Raycast show the old scene.
        const Scene& GetScene() const { return *m_ActiveScene; }
        // Adds an entity that is saved with the scene. Its asset path is
        // relative to directory, unless it's absolute.
        // One without an asset is a marker: a named place that draws nothing.
        std::optional<EntityId> AddEntity(const EntityDescription& entity, const std::filesystem::path& directory = {});
        // The first saved entity with this name, such as a marker the scene
        // places for the game. Look it up again after the scene is edited
        // or reloaded: ids don't survive that.
        std::optional<EntityId> FindEntity(std::string_view name) const;
        void RemoveInstance(EntityId instance);
        void SetInstanceStatic(EntityId instance, bool isStatic);
        void SetInstanceName(EntityId instance, const std::string& name);
        // The scene as it would be saved, with absolute asset paths
        SceneDescription DescribeScene() const;
        // Swaps the scene's saved entities for these (asset paths absolute),
        // e.g. to undo. Returns the new ids in order, 0 for any that failed to load.
        std::vector<EntityId> ReplaceEntities(const std::vector<EntityDescription>& entities);
        // Replaces the saved entities, camera and lighting with a scene file's
        // (or a single glTF file's). False, changing nothing, if it can't be read.
        bool OpenScene(const std::filesystem::path& path);
        void NewScene();
        // The file SaveScene last wrote or the scene was opened from (for a
        // glTF file, a scene file next to it)
        const std::filesystem::path& GetScenePath() const { return m_SavePath; }
        // Where the view starts when the scene is loaded
        const SceneCameraDescription& GetSceneCamera() const { return m_SceneCamera; }
        void SetSceneCamera(glm::vec3 position, glm::vec3 target) { m_SceneCamera = { position, target }; }
        bool IsStaticGeometryCurrent() const { return !m_StaticGeometryChanged && !m_PendingRayTracingGeometry.valid(); }

        // In editing mode game layers stop updating, the camera is left to
        // the editor, the cursor is released and Escape no longer quits
        void SetEditing(bool editing);
        bool IsEditing() const { return m_Editing; }
        void SetCursorCaptured(bool captured);
        void SetSettingsVisible(bool visible) { m_ShowSettings = visible; }
        bool IsSettingsVisible() const { return m_ShowSettings; }
        void Quit() { m_Running = false; }

        // The nearest instance, static or not, along a ray through the
        // scene's triangles as they are right now. Meant for a click, not for
        // every frame: it checks every triangle of each mesh the ray could hit.
        // With savedOnly, instances game code added (a character) are ignored.
        std::optional<PickHit> PickInstance(glm::vec3 origin, glm::vec3 direction, bool savedOnly = false) const;
        // In the model's own space; move the corners by an instance's transform
        Bounds GetModelBounds(Assets::ModelHandle model) const;
        // Clip space with Vulkan's conventions: y down, depth 0 to 1
        const glm::mat4& GetProjectionMatrix() const { return m_Renderer.GetProjectionMatrix(); }
        
        void Run();
        void Shutdown();
    private:
        ApplicationSpecification m_Spec{};
        bool m_Running = true;
        bool m_IsShutdown = false;
        float m_TotalTime = 0.0f;
        float m_CubeRotationSpeed = 1.0f; // radians per second (Should be moved to the scene or entity)
        
        uint64_t m_FrameCount = 0;
        Input::InputState m_InputState;
        Renderer::RenderMode m_RenderMode;
        bool m_ToggleRenderModeHeld = false; // Toggle once per press, not every frame the key is down

        // Settings panel (Dear ImGui), shown with Tab; the cursor is released while it's open
        bool m_ShowSettings = false;
        bool m_ToggleSettingsHeld = false;
        Renderer::SceneLighting m_Lighting;
        bool m_FreeCamera = true;
        bool m_Editing = false;
        SceneCameraDescription m_SceneCamera{ .position = { 0.0f, 2.0f, 0.0f }, .target = { 0.0f, 2.0f, -1.0f } };
        bool m_StaticGeometryChanged = false;
        std::chrono::steady_clock::time_point m_StaticRebuildAt{};
        void MarkStaticGeometryChanged();
        void RebuildStaticGeometryIfChanged();
        void RemoveSavedInstances();
        void RefreshLightsAndShadowBounds();
        std::vector<Assets::PointLight> m_SceneLights; // As the entities' assets give them
        bool m_LightsCollected = false;
        mutable std::unordered_map<Assets::MeshHandle, Bounds> m_MeshBounds; // In each mesh's own space
        const Bounds& GetMeshBounds(Assets::MeshHandle mesh) const;
        // CPU copy of the static scene's BVH and triangles, for Raycast
        struct CollisionGeometry
        {
            RayTracing::WideBvh bvh;
            std::vector<RayTracing::TrianglePositions> triangles;
        };
        std::optional<CollisionGeometry> m_Collision;
        Renderer::SceneLighting m_LoadedLighting; // Defaults plus the scene's own sun and sky; "Reset lighting" returns here
        Renderer::ToonSettings m_ToonSettings;
        void DrawSettingsPanel();

        // Light editor: the scene's point lights and glowing (emissive)
        // materials, as edited and as loaded. Colors and strengths are split
        // so a slider can scale a light without changing its hue.
        struct EditablePointLight
        {
            std::string name;
            glm::vec3 position;
            glm::vec3 color;
            float intensity; // W/sr: a surface at distance d facing the light receives color * intensity / d^2
        };
        struct EditableGlow
        {
            Assets::MaterialHandle handle;
            std::string name;
            glm::vec3 color;
            float strength;
        };
        std::vector<EditablePointLight> m_PointLights, m_LoadedPointLights;
        std::vector<EditableGlow> m_Glows, m_LoadedGlows;
        void CollectEditableLights(const std::vector<Assets::PointLight>& pointLights);
        void ApplyGlow(const EditableGlow& glow);
        void DrawLightsSection();

        std::unique_ptr<IWindow> m_Window;
        Renderer::VulkanRenderer m_Renderer; // Maybe this can be abstracted later
        std::vector<std::unique_ptr<Layer>> m_Layers;
        std::unique_ptr<Assets::IModelLoader> m_ModelLoader;
        Assets::AssetRegistry m_AssetRegistry;

        // Scene data
        std::optional<Scene> m_ActiveScene; 
        std::optional<Camera> m_Camera;
        // std::optional<Renderer::VulkanModelData> m_Mesh; // TODO: Should be a full scene object as active scene

        Renderer::MeshGPUCache m_MeshCache;
        Renderer::TextureGPUCache m_TextureCache;
        Renderer::MaterialGPUCache m_MaterialCache;

        // Ray tracing geometry being built on a worker thread. The worker reads
        // m_AssetRegistry, so the registry must not be modified while this is
        // valid. Declared after the registry so it is destroyed (and waited on) first.
        std::future<RayTracing::SceneGeometry> m_PendingRayTracingGeometry;

        std::optional<Benchmark> m_Benchmark;

        Renderer::VulkanModelData LoadMesh(const std::filesystem::path &path);
        std::optional<Renderer::DrawFrameTime> Draw(float dt);
        bool LoadScene(const std::filesystem::path &path);
        void ApplySceneLighting(const std::optional<Assets::SunSource>& sun, const Assets::SkySource& sky);
        void DrawSceneSection();

        // Every glTF file loaded, by absolute path, with the camera, sun and
        // sky it brought (the registry keeps only the meshes, materials and lights)
        struct LoadedAsset
        {
            std::filesystem::path path;
            std::optional<Assets::CameraSource> camera;
            std::optional<Assets::SunSource> sun;
            Assets::SkySource sky;
            Assets::ModelHandle handle;
        };
        std::map<std::filesystem::path, LoadedAsset> m_LoadedAssets;
        const LoadedAsset* LoadAsset(const std::filesystem::path& path);
        // Where the settings panel saves the scene: the scene file it came
        // from, or one next to the glTF file it was
        std::filesystem::path m_SavePath;
        std::array<char, 1024> m_SavePathInput{};
        std::string m_SaveStatus;
        void StartRayTracingGeometryBuild();
        void UploadRayTracingGeometryIfReady();
        void LoadDefaultAssets();
    };
}
