#pragma once
#include <filesystem>
#include <string>
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
        // glTF file to load, relative to the binary dir. Its first camera, if
        // it has one, is where the view starts
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
        // With the free camera off, the camera stays wherever a layer puts it
        // instead of flying around with WASD and the mouse
        void SetFreeCamera(bool enabled) { m_FreeCamera = enabled; }
        // Loads a glTF model into the asset registry without placing it
        std::optional<Assets::ModelHandle> LoadModel(const std::filesystem::path& path);
        // Places a model in the active scene; returns the instance's index for
        // SetInstanceTransform. Only dynamic instances may be added after the
        // scene has loaded (static ones would miss the ray tracer's BVH).
        size_t AddDynamicInstance(Assets::ModelHandle model, const glm::mat4& transform);
        void SetInstanceTransform(size_t instance, const glm::mat4& transform);
        // The first hit along a ray (direction of unit length) against the
        // static scene, within maxDistance. The collision geometry is built
        // with the ray tracer's BVH on a worker thread, so there are no hits
        // until IsCollisionReady(). Alpha masked surfaces (leaves) count as solid.
        std::optional<RaycastHit> Raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance) const;
        bool IsCollisionReady() const { return m_Collision.has_value(); }
        
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
        void CollectEditableLights(Assets::ModelHandle model);
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
        void LoadScene(const std::filesystem::path &path);
        void ApplySceneLighting(const Assets::ModelSource& model);
        void StartRayTracingGeometryBuild();
        void UploadRayTracingGeometryIfReady();
        void LoadDefaultAssets();
    };
}
