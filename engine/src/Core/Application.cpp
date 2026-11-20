#include "Momo/Core/Application.h"
#include "Momo/Profiling/Timer.h"
#include "Momo/Profiling/FrameStats.h"
#include "Momo/Cache/DataCache.h"
#include "Momo/Renderer/VulkanMeshData.h"
#include "Momo/Renderer/VulkanRenderer.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <Momo/Logging/Logger.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
namespace Momo
{
    Application::Application(const ApplicationSpecification& spec)
        : m_Spec(spec),
          m_RenderMode(spec.InitialRenderMode),
          m_MeshCache(m_AssetRegistry, {m_Renderer, nullptr}), // Mesh cache has no further dependencies
          m_TextureCache(m_AssetRegistry, {m_Renderer, nullptr}), // Texture cache has no further dependencies
          m_MaterialCache(m_AssetRegistry,  {m_Renderer, &m_TextureCache})
    {
        LOG_INFO("Momo", "Starting '{}' ({}x{})", m_Spec.Name, m_Spec.WindowSpec.Width, m_Spec.WindowSpec.Height);
        if (m_Spec.Benchmark)
        {
            m_Benchmark.emplace(*m_Spec.Benchmark);
            if (m_Spec.ScenePath != ApplicationSpecification{}.ScenePath)
                LOG_WARN("Momo", "Benchmark views are placed in Sponza; scene {} will be measured from them anyway", m_Spec.ScenePath.generic_string());
            m_RenderMode = Renderer::RenderMode::RayTraced;
        }
        if (m_Spec.Screenshot)
            m_RenderMode = Renderer::RenderMode::RayTraced;
        m_Window = std::unique_ptr<IWindow>(IWindow::Create(m_Spec.WindowSpec));
        m_Renderer.Init(*m_Window);
        // Benchmarks have no UI
        if (!m_Spec.Benchmark)
        {
            ImGui::CreateContext();
            ImGui::GetIO().IniFilename = nullptr; // Don't leave imgui.ini files around
            ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(m_Window->GetNativeHandle()), true);
            m_Renderer.InitImGui();
            LOG_INFO("Momo", "Press Tab for render settings, P to cycle render modes");
        }
        m_ModelLoader = std::unique_ptr<Assets::IModelLoader>(Assets::IModelLoader::CreateGltfModelLoader());
        m_AssetRegistry.Init();
        LoadScene(m_Spec.ScenePath);
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
        const std::filesystem::path& modelPath = path;

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
        // Models have no transform of their own yet, so the camera's model space is world space
        if (modelSource->camera.has_value()) {
            m_Camera = Camera::LookingAlong(modelSource->camera->position, glm::vec3(0.0f, 1.0f, 0.0f), modelSource->camera->forward);
        } else {
            m_Camera = Camera(glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90.0f, 0.0f);
        }
        ApplySceneLighting(*modelSource);
        Assets::ModelHandle handle = m_AssetRegistry.RegisterModel(std::move(*modelSource));
        m_ActiveScene->AddModel(handle);
        // Models have no transform of their own yet, so model space is world space
        m_Renderer.SetPointLights(m_AssetRegistry.Get(handle).pointLights);
        CollectEditableLights(handle);
        glm::vec3 boundsMin(std::numeric_limits<float>::max());
        glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
        for (const Assets::MeshHandle& meshHandle : m_AssetRegistry.Get(handle).meshes)
        {
            const Assets::Mesh& mesh = m_AssetRegistry.Get(meshHandle);
            for (const Geometry::Vertex& vertex : mesh.meshData.vertices)
            {
                glm::vec3 position = glm::vec3(mesh.localTransform * glm::vec4(vertex.position, 1.0f));
                boundsMin = glm::min(boundsMin, position);
                boundsMax = glm::max(boundsMax, position);
            }
        }
        if (boundsMin.x <= boundsMax.x)
            m_Renderer.SetShadowBounds(boundsMin, boundsMax);
        StartRayTracingGeometryBuild();
    }

    // A scene without its own sun or sky keeps the defaults (a sunset)
    void Application::ApplySceneLighting(const Assets::ModelSource& model)
    {
        Renderer::SceneLighting lighting{};
        if (model.sun)
        {
            // Inverse of SceneLighting::SunDirection()
            glm::vec3 toward = model.sun->towardSun;
            lighting.sunElevation = glm::degrees(std::asin(std::clamp(toward.y, -1.0f, 1.0f)));
            lighting.sunAzimuth = glm::degrees(std::atan2(toward.z, toward.x));
            lighting.sunColor = model.sun->color;
            lighting.sunIntensity = model.sun->intensity;
        }
        lighting.skyZenith = model.sky.zenith.value_or(lighting.skyZenith);
        lighting.skyHorizon = model.sky.horizon.value_or(lighting.skyHorizon);
        lighting.skyGround = model.sky.ground.value_or(lighting.skyGround);
        lighting.exposure = model.sky.exposure.value_or(lighting.exposure);
        m_Lighting = m_LoadedLighting = lighting;
    }

    // The BVH build takes seconds in debug builds. Running it on the main
    // thread before the first frame kept the window from processing events
    // long enough for the window manager to flag it as not responding.
    void Application::StartRayTracingGeometryBuild()
    {
        // The scene is copied (it's a list of handles), so the worker never
        // touches m_ActiveScene
        m_PendingRayTracingGeometry = std::async(std::launch::async,
            [&registry = m_AssetRegistry, scene = *m_ActiveScene]
            {
                Profiling::Timer bvhTimer;
                bvhTimer.Start();
                RayTracing::SceneGeometry geometry = RayTracing::BuildSceneGeometry(registry, scene);
                bvhTimer.End();
                const RayTracing::WideBvhStats& bvhStats = geometry.bvh.stats;
                LOG_INFO("Momo", "Built BVH over {} triangles in {:.1f} ms: {} wide nodes, depth {} (binary tree: {} nodes, {} leaves, depth {}, SAH cost {:.1f})",
                         geometry.positions.size(), bvhTimer.GetElapsedTimeInMs(), bvhStats.nodeCount, bvhStats.maxDepth,
                         bvhStats.binary.nodeCount, bvhStats.binary.leafCount, bvhStats.binary.maxDepth, bvhStats.binary.sahCost);
                return geometry;
            });
    }

    // Called once per frame. Vulkan work stays on the main thread; only the CPU build runs on the worker.
    void Application::UploadRayTracingGeometryIfReady()
    {
        if (!m_PendingRayTracingGeometry.valid() ||
            m_PendingRayTracingGeometry.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        {
            return;
        }
        // get() rethrows anything the build threw. The renderer keeps only the GPU copy
        RayTracing::SceneGeometry geometry = m_PendingRayTracingGeometry.get();

        // The textures are shared with the raster path, which has usually uploaded them already
        std::vector<vk::ImageView> textureViews;
        textureViews.reserve(geometry.textures.size());
        for (Assets::TextureHandle texture : geometry.textures)
            textureViews.push_back(*m_TextureCache.GetOrCreate(texture).imageView);

        m_Renderer.UploadRayTracingScene(std::move(geometry), std::move(textureViews));
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

            UploadRayTracingGeometryIfReady();

            bool toggleRenderMode = m_InputState.keys.IsKeyPressed(Input::Key::P);
            if (toggleRenderMode && !m_ToggleRenderModeHeld && !m_Benchmark && !m_Spec.Screenshot)
            {
                // Raster -> toon -> ray traced -> raster
                switch (m_RenderMode)
                {
                case Renderer::RenderMode::Raster:
                    m_RenderMode = Renderer::RenderMode::Toon;
                    break;
                case Renderer::RenderMode::Toon:
                    if (m_Renderer.IsRayTracingAvailable())
                        m_RenderMode = Renderer::RenderMode::RayTraced;
                    else
                    {
                        if (m_PendingRayTracingGeometry.valid())
                            LOG_INFO("Momo", "Ray tracing geometry is still building; skipping ray traced mode.");
                        else
                            LOG_WARN("Momo", "Ray tracing is unavailable; skipping ray traced mode.");
                        m_RenderMode = Renderer::RenderMode::Raster;
                    }
                    break;
                case Renderer::RenderMode::RayTraced:
                    m_RenderMode = Renderer::RenderMode::Raster;
                    break;
                }
                LOG_INFO("Momo", "Render mode: {}", Renderer::RenderModeName(m_RenderMode));
            }
            m_ToggleRenderModeHeld = toggleRenderMode;

            bool toggleSettings = m_InputState.keys.IsKeyPressed(Input::Key::Tab);
            if (toggleSettings && !m_ToggleSettingsHeld && !m_Benchmark)
            {
                m_ShowSettings = !m_ShowSettings;
                m_Window->SetCursorCaptured(!m_ShowSettings);
            }
            m_ToggleSettingsHeld = toggleSettings;

            if (!m_Benchmark)
            {
                // A frame is built even while the panel is hidden, so stale draw data never lingers
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                if (m_ShowSettings)
                    DrawSettingsPanel();
                ImGui::Render();
                m_Renderer.SetLighting(m_Lighting);
                m_Renderer.SetToonSettings(m_ToonSettings);
                std::vector<Assets::PointLight> pointLights;
                for (const EditablePointLight& light : m_PointLights)
                    pointLights.push_back({ .position = light.position, .intensity = light.color * light.intensity, .name = light.name });
                m_Renderer.SetPointLights(pointLights);
            }

            updateTimer.Start();
            for (auto& layer : m_Layers)
                layer->OnUpdate(dt);

            // Benchmarks place the camera themselves and ignore input; screenshots keep the scene's camera
            if (!m_Benchmark && !m_Spec.Screenshot)
                m_Camera->OnUpdate(dt, m_InputState);
            else if (!m_Benchmark->IsFinished())
            {
                m_Camera->SetPose(m_Benchmark->CurrentView().Position, m_Benchmark->CurrentView().Yaw, m_Benchmark->CurrentView().Pitch);
                m_Renderer.SetTraversalCounting(m_Benchmark->IsCounting());
                m_Renderer.SetRayCapture(m_Benchmark->IsCapturing());
            }
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

            // Frames before the ray tracing geometry is uploaded render raster and don't count
            if (m_Benchmark && drawFrameTime.has_value() && m_Renderer.IsRayTracingAvailable())
            {
                if (std::optional<size_t> finishedView = m_Benchmark->Record(*drawFrameTime, frameStats))
                {
                    vk::Extent2D extent = m_Renderer.GetSwapchainExtent();
                    m_Benchmark->SetTraversalCounts(*finishedView, m_Renderer.ReadAndResetTraversalCounters(),
                                                    uint64_t(extent.width) * extent.height);
                    m_Renderer.SetTraversalCounting(false);
                    m_Benchmark->SetTraceOnlyResults(*finishedView, m_Renderer.RunTraceOnlyExperiment());
                    m_Renderer.SetRayCapture(false);
                    if (m_Renderer.SaveRayTracingOutput(m_Benchmark->ScreenshotPath(*finishedView)))
                        m_Benchmark->SetScreenshotSaved(*finishedView);
                }
                if (m_Benchmark->IsFinished())
                {
                    vk::Extent2D extent = m_Renderer.GetSwapchainExtent();
                    const Renderer::RayTracingSceneBuffers& scene = *m_Renderer.GetRayTracingScene();
                    m_Benchmark->WriteResults({
                        .Gpu = m_Renderer.GetDeviceName(),
                        .Width = extent.width,
                        .Height = extent.height,
                        .Bounces = m_Renderer.GetRayTracingMaxBounces(),
                        .Bvh = scene.stats,
                        .Triangles = scene.triangleCount,
                        .GpuBytes = scene.gpuBytes,
                        .NodeBytes = sizeof(RayTracing::CompressedWideNode),
                    });
                    m_Running = false;
                }
            }

            if (m_Spec.Screenshot && m_Renderer.IsRayTracingAvailable() &&
                m_Renderer.GetAccumulatedSamples() >= m_Spec.Screenshot->Samples)
            {
                if (!m_Renderer.SaveRayTracingOutput(m_Spec.Screenshot->OutputPath))
                    LOG_ERROR("Momo", "Couldn't save the screenshot");
                m_Running = false;
            }

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

    namespace
    {
        // value = color * strength, with the strength taken from the brightest channel
        void SplitColor(const glm::vec3& value, glm::vec3& color, float& strength)
        {
            strength = std::max(value.r, std::max(value.g, value.b));
            color = strength > 0.0f ? value / strength : glm::vec3(1.0f);
        }
    }

    void Application::CollectEditableLights(Assets::ModelHandle model)
    {
        m_PointLights.clear();
        for (const Assets::PointLight& light : m_AssetRegistry.Get(model).pointLights)
        {
            EditablePointLight editable{ .name = light.name, .position = light.position };
            SplitColor(light.intensity, editable.color, editable.intensity);
            m_PointLights.push_back(editable);
        }
        m_LoadedPointLights = m_PointLights;

        m_Glows.clear();
        for (const Assets::MeshHandle& meshHandle : m_AssetRegistry.Get(model).meshes)
        {
            Assets::MaterialHandle materialHandle = m_AssetRegistry.Get(meshHandle).materialHandle;
            const Assets::Material& material = m_AssetRegistry.Get(materialHandle);
            bool known = std::any_of(m_Glows.begin(), m_Glows.end(), [&](const EditableGlow& glow) { return glow.handle == materialHandle; });
            if (known || material.emissive == glm::vec3(0.0f))
                continue;
            EditableGlow glow{ .handle = materialHandle, .name = material.name };
            SplitColor(material.emissive, glow.color, glow.strength);
            m_Glows.push_back(glow);
        }
        m_LoadedGlows = m_Glows;
    }

    void Application::ApplyGlow(const EditableGlow& glow)
    {
        Assets::Material material = m_AssetRegistry.Get(glow.handle);
        material.emissive = glow.color * glow.strength;
        m_Renderer.UpdateMaterial(glow.handle, m_MaterialCache.GetOrCreate(glow.handle), material);
    }

    void Application::DrawLightsSection()
    {
        if (!ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen))
            return;
        ImGui::PushID("Lights");
        if (m_PointLights.empty() && m_Glows.empty())
            ImGui::TextDisabled("The scene has no point lights or glowing materials");

        for (size_t i = 0; i < m_PointLights.size(); i++)
        {
            EditablePointLight& light = m_PointLights[i];
            ImGui::PushID(static_cast<int>(i));
            std::string label = light.name.empty() ? "Point light " + std::to_string(i) : light.name;
            ImGui::SeparatorText(label.c_str());
            ImGui::DragFloat3("Position", &light.position.x, 0.05f);
            ImGui::ColorEdit3("Color", &light.color.x);
            ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 2000.0f, "%.0f W/sr", ImGuiSliderFlags_Logarithmic);
            ImGui::PopID();
        }

        for (size_t i = 0; i < m_Glows.size(); i++)
        {
            EditableGlow& glow = m_Glows[i];
            ImGui::PushID(static_cast<int>(m_PointLights.size() + i));
            std::string label = (glow.name.empty() ? "Material " + std::to_string(glow.handle.id) : glow.name) + " (glow)";
            ImGui::SeparatorText(label.c_str());
            bool changed = ImGui::ColorEdit3("Color", &glow.color.x);
            changed |= ImGui::SliderFloat("Strength", &glow.strength, 0.0f, 50.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            if (changed)
                ApplyGlow(glow);
            ImGui::PopID();
        }

        if (ImGui::Button("Reset lights"))
        {
            m_PointLights = m_LoadedPointLights;
            m_Glows = m_LoadedGlows;
            for (const EditableGlow& glow : m_Glows)
                ApplyGlow(glow);
        }
        ImGui::PopID();
    }

    void Application::DrawSettingsPanel()
    {
        ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
        ImGui::Begin("Render settings (Tab to close)");

        int mode = static_cast<int>(m_RenderMode);
        ImGui::RadioButton("Raster", &mode, static_cast<int>(Renderer::RenderMode::Raster));
        ImGui::SameLine();
        ImGui::RadioButton("Toon", &mode, static_cast<int>(Renderer::RenderMode::Toon));
        ImGui::SameLine();
        ImGui::BeginDisabled(!m_Renderer.IsRayTracingAvailable());
        ImGui::RadioButton("Ray traced", &mode, static_cast<int>(Renderer::RenderMode::RayTraced));
        ImGui::EndDisabled();
        if (mode != static_cast<int>(m_RenderMode))
        {
            m_RenderMode = static_cast<Renderer::RenderMode>(mode);
            LOG_INFO("Momo", "Render mode: {}", Renderer::RenderModeName(m_RenderMode));
        }
        ImGui::Text("%.1f ms/frame (%.0f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        if (ImGui::CollapsingHeader("Lighting", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::PushID("Lighting");
            Renderer::SceneLighting& light = m_Lighting;
            ImGui::SliderFloat("Sun azimuth", &light.sunAzimuth, -180.0f, 180.0f, "%.0f deg");
            ImGui::SliderFloat("Sun elevation", &light.sunElevation, -10.0f, 90.0f, "%.0f deg");
            ImGui::ColorEdit3("Sun color", &light.sunColor.x);
            ImGui::SliderFloat("Sun intensity", &light.sunIntensity, 0.0f, 20.0f);
            ImGui::ColorEdit3("Sky zenith", &light.skyZenith.x);
            ImGui::ColorEdit3("Sky horizon", &light.skyHorizon.x);
            ImGui::ColorEdit3("Sky ground", &light.skyGround.x);
            ImGui::SliderFloat("Exposure", &light.exposure, 0.05f, 3.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            ImGui::SeparatorText("Shadows (raster and toon)");
            ImGui::Checkbox("Shadows", &light.shadows);
            ImGui::SliderFloat("Shadow softness", &light.shadowSoftness, 0.0f, 4.0f, "%.1f texels");
            ImGui::SliderFloat("Shadow bias", &light.shadowNormalBias, 0.0f, 6.0f, "%.1f texels");
            ImGui::Checkbox("Point light shadows", &light.pointShadows);
            ImGui::SliderFloat("Point shadow bias", &light.pointShadowBias, 0.0f, 6.0f, "%.1f texels");
            ImGui::Checkbox("Ambient occlusion", &light.ambientOcclusion);
            ImGui::SliderFloat("Occlusion radius", &light.occlusionRadius, 0.1f, 4.0f, "%.2f");
            ImGui::SliderFloat("Occlusion strength", &light.occlusionStrength, 0.25f, 4.0f, "%.2f");
            ImGui::Checkbox("Show occlusion only", &light.showOcclusion);
            if (ImGui::Button("Reset lighting"))
                light = m_LoadedLighting;
            ImGui::PopID();
        }

        DrawLightsSection();

        if (ImGui::CollapsingHeader("Toon shading", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::PushID("Toon");
            Renderer::ToonSettings& toon = m_ToonSettings;
            ImGui::SeparatorText("Bands");
            ImGui::SliderInt("Bands", &toon.bands, 2, 8);
            ImGui::SliderFloat("Band softness", &toon.bandSoftness, 0.0f, 1.0f);
            ImGui::ColorEdit3("Shadow tint", &toon.shadowTint.x);
            ImGui::SeparatorText("Highlights");
            ImGui::SliderFloat("Specular strength", &toon.specularStrength, 0.0f, 3.0f);
            ImGui::SliderFloat("Specular size", &toon.specularSize, 0.0f, 1.0f);
            ImGui::ColorEdit3("Rim color", &toon.rimColor.x);
            ImGui::SliderFloat("Rim strength", &toon.rimStrength, 0.0f, 3.0f);
            ImGui::SliderFloat("Rim width", &toon.rimWidth, 0.0f, 1.0f);
            ImGui::SeparatorText("Outlines");
            ImGui::SliderFloat("Outline width", &toon.outlineWidth, 0.0f, 6.0f, "%.1f px");
            ImGui::ColorEdit3("Outline color", &toon.outlineColor.x);
            ImGui::SliderFloat("Depth threshold", &toon.depthThreshold, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Normal threshold", &toon.normalThreshold, 0.01f, 2.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::Button("Reset toon"))
                toon = Renderer::ToonSettings{};
            ImGui::PopID();
        }
        ImGui::End();
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

        Renderer::DrawFrameTime renderFrameTime = m_Renderer.RenderFrame(vulkanModelData, m_Camera->GetViewMatrix(), modelMatrix, m_RenderMode);
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

        // Shutdown renderer first (destroys Vulkan instance before GLFW terminates).
        // It shuts down the ImGui Vulkan backend, which needs the context still alive.
        m_Renderer.Shutdown();
        if (ImGui::GetCurrentContext())
        {
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
        }

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
