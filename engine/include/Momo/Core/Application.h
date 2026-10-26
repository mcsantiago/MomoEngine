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


namespace Momo
{
    struct ApplicationSpecification
    {
        std::string Name = "Untitled";
        WindowSpecification WindowSpec{};
        Renderer::RenderMode InitialRenderMode = Renderer::RenderMode::Raster; // Toggled at runtime with P
        // glTF file to load, relative to the binary dir. Its first camera, if
        // it has one, is where the view starts
        std::filesystem::path ScenePath = "Assets/Models/Sponza/Sponza.gltf";
        // Set to run the ray tracing benchmark instead of interactive mode; the app exits when done
        std::optional<BenchmarkSpecification> Benchmark;
    };

    class Layer
    {
    public:
        virtual ~Layer() = default;
        virtual void OnAttach() {}
        virtual void OnDetach() {}
        virtual void OnUpdate(float dt) {}
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
            m_Layers.emplace_back(std::move(layer));
            raw->OnAttach();
            return raw;
        }
        
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
        void StartRayTracingGeometryBuild();
        void UploadRayTracingGeometryIfReady();
        void LoadDefaultAssets();
    };
}
