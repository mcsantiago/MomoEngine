#include "Momo/Core/Application.h"
#include "Momo/Profiling/Timer.h"
#include "Momo/Profiling/FrameStats.h"
#include "Momo/Cache/DataCache.h"
#include "Momo/Renderer/VulkanMeshData.h"
#include "Momo/Renderer/VulkanRenderer.h"
#include <chrono>
#include <Momo/Logging/Logger.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
namespace Momo
{
    Application::Application(const ApplicationSpecification& spec)
        : m_Spec(spec),
          m_MeshCache(m_AssetRegistry, {m_Renderer, nullptr}), // Mesh cache has no further dependencies
          m_TextureCache(m_AssetRegistry, {m_Renderer, nullptr}), // Texture cache has no further dependencies
          m_MaterialCache(m_AssetRegistry,  {m_Renderer, &m_TextureCache})
    {
        LOG_INFO("Momo", "Starting '{}' ({}x{})", m_Spec.Name, m_Spec.WindowSpec.Width, m_Spec.WindowSpec.Height);
        m_Window = std::unique_ptr<IWindow>(IWindow::Create(m_Spec.WindowSpec));
        m_Renderer.Init(*m_Window);
        m_ModelLoader = std::unique_ptr<Assets::IModelLoader>(Assets::IModelLoader::CreateGltfModelLoader());
        m_AssetRegistry.Init();
        LoadScene(""); // TODO: Replace with actual scene path
    }

    Application::~Application()
    {
        // Ensure shutdown is called even if the user forgot
        if (!m_IsShutdown)
        {
            LOG_WARN("Momo", "Shutdown() was not called explicitly. Calling now...");
            
            Shutdown();
        }
    }

    void Application::LoadScene(const std::filesystem::path &path)
    {
        // const std::filesystem::path modelPath = "Assets/Models/Duck/Duck.gltf";
        // const std::filesystem::path modelPath = "Assets/Models/Panko/PANKO_Rigged.glb";
        // const std::filesystem::path modelPath = "Assets/Models/FlightHelmet/FlightHelmet.gltf";
        const std::filesystem::path modelPath = "Assets/Models/Sponza/Sponza.gltf";

        if (!std::filesystem::exists(modelPath)) {
            LOG_ERROR("Momo", "Model file does not exist: {}", modelPath.generic_string());
            return;
        }
        m_ActiveScene = Scene();
        auto modelSource = m_ModelLoader->LoadModel(modelPath);
        if (!modelSource.has_value()) {
            LOG_ERROR("Momo", "Failed to load model from path: {}", modelPath.generic_string());
            return;
        }
        Assets::ModelHandle handle = m_AssetRegistry.RegisterModel(std::move(*modelSource));
        m_ActiveScene->AddModel(handle);
        m_Camera = Camera(glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
    }

    void Application::Run()
    {
        using clock = std::chrono::steady_clock;
        auto last = clock::now();
        auto frameStats = Profiling::FrameTime{};
        Profiling::FrameStatsAccumulator statsAccumulator;
        auto statsWindowStart = last;
        Profiling::Timer updateTimer;


        while (m_Running && !m_Window->ShouldClose())
        {
            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - last).count(); // default to seconds
            m_TotalTime += dt;
            last = now;

            m_Window->PollEvents();
            m_InputState = m_Window->ReadInput();
            if (m_InputState.keys.IsKeyPressed(Input::Key::Escape))
            {
                LOG_INFO("Momo", "Quit requested. Exiting main loop.");
                m_Running = false;
                break;
            }

            updateTimer.Start();
            for (auto& layer : m_Layers)
                layer->OnUpdate(dt);

            m_Camera->OnUpdate(dt, m_InputState);
            updateTimer.End();

            auto drawFrameTime = Draw(dt);

            frameStats.frame        = m_FrameCount;
            frameStats.cpuTotalMs   = std::chrono::duration<float, std::milli>(clock::now() - now).count();
            frameStats.updateMs     = updateTimer.GetElapsedTimeInMs();
            frameStats.submitMs     = drawFrameTime.has_value() ? drawFrameTime->submitMs : 0.0f;
            frameStats.recordMs     = drawFrameTime.has_value() ? drawFrameTime->recordMs : 0.0f;
            frameStats.fenceWaitMs  = drawFrameTime.has_value() ? drawFrameTime->fenceWaitMs : 0.0f;
            frameStats.gpuMs        = drawFrameTime.has_value() ? drawFrameTime->gpuFrameMs : 0.0f;
            statsAccumulator.Add(frameStats);

            // Log a summary roughly once per second instead of every frame
            double statsWindowSeconds = std::chrono::duration<double>(clock::now() - statsWindowStart).count();
            if (statsWindowSeconds >= 1.0)
            {
                statsAccumulator.LogAndReset(statsWindowSeconds);
                statsWindowStart = clock::now();
            }

            m_FrameCount++;
        }
    }

    std::optional<Renderer::DrawFrameTime> Application::Draw(float dt)
    {
        // TODO: Should be a dedicated entity.OnUpdate(dt) call instead of directly manipulating the model matrix here
        // Construct VulkanModelData for each scene
        if (!m_ActiveScene.has_value()) {
            LOG_ERROR("Momo", "No active scene loaded.");
            return std::nullopt;
        }

        // TODO: This doesn't have to be done every frame, but for now it's a simple way to ensure the data is up-to-date
        Renderer::VulkanModelData vulkanModelData{};
        const Assets::ModelHandle& modelHandle = m_ActiveScene->GetModels().front(); // TODO: Handle multiple models and empty scene cases
        for (const auto& meshHandle : m_AssetRegistry.Get(modelHandle).meshes)
        {
            const Assets::Mesh& mesh = m_AssetRegistry.Get(meshHandle);
            const Renderer::VulkanMaterialData& materialData = m_MaterialCache.GetOrCreate(mesh.materialHandle);

            vulkanModelData.meshes.push_back(Renderer::VulkanMeshData {
                .gpuMesh = m_MeshCache.GetOrCreate(meshHandle),
                .localTransform = mesh.localTransform,
                .materialData = materialData
            });
        }
        //glm::vec3 rotationAxis = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f));
        //glm::mat4 modelMatrix = glm::rotate(glm::mat4(1.0f), m_TotalTime * m_CubeRotationSpeed, rotationAxis);
        glm::mat4 modelMatrix = glm::mat4(1.0f); // Identity for now, should be replaced with actual entity transform
        LOG_TRACE("Momo", "Model matrix: {}", glm::to_string(modelMatrix));

        Renderer::DrawFrameTime renderFrameTime = m_Renderer.RenderFrame(vulkanModelData, m_Camera->GetViewMatrix(), modelMatrix);
        return renderFrameTime;
    }

    void Application::Shutdown()
    {
        if (m_IsShutdown)
        {
            LOG_WARN("Momo", "Shutdown() called multiple times. Ignoring...");
            return;
        }

        LOG_INFO("Momo", "Shutting down...");

        // Shutdown renderer first (destroys Vulkan instance before GLFW terminates)
        m_Renderer.Shutdown();

        // Detach all layers before they are destroyed
        for (auto& layer : m_Layers)
        {
            layer->OnDetach();
        }

        m_IsShutdown = true;
        LOG_INFO("Momo", "Shutdown complete.");
    }

    Renderer::VulkanModelData Application::LoadMesh(const std::filesystem::path &path)
    {
        if (!m_ModelLoader)
        {
            LOG_ERROR("Momo", "Model loader not initialized.");
            throw std::runtime_error("Model loader not initialized.");
        }

        auto modelSource = m_ModelLoader->LoadModel(path);
        if (!modelSource)
        {
            LOG_ERROR("Momo", "Failed to load model from path: {}", path.string());
            throw std::runtime_error("Failed to load model from path: " + path.string());
        }

        // Local indices become engine-wide handles here; everything below reads
        // the registry rather than the loader's output.
        Assets::ModelHandle modelHandle = m_AssetRegistry.RegisterModel(std::move(*modelSource));

        try {
            Renderer::VulkanModelData modelData;
            for (const auto& meshHandle : m_AssetRegistry.Get(modelHandle).meshes)
            {
                const Assets::Mesh& mesh = m_AssetRegistry.Get(meshHandle);
                const Assets::Material& material = m_AssetRegistry.Get(mesh.materialHandle);

                m_MeshCache.GetOrCreate(meshHandle);
                m_MaterialCache.GetOrCreate(mesh.materialHandle);
                m_TextureCache.GetOrCreate(material.baseColorTextureHandle);
            }
            return modelData;
        } catch (const std::exception& e) {
            LOG_ERROR("Momo", "Exception occurred while loading mesh: {}", e.what());
            throw;
        }
    }
} // namespace Momo
