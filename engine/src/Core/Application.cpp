#include "Momo/Core/Application.h"
#include "Momo/Profiling/Timer.h"
#include "Momo/Profiling/FrameStats.h"
#include "Momo/Cache/DataCache.h"
#include "Momo/Renderer/VulkanMeshData.h"
#include "Momo/Renderer/VulkanRenderer.h"
#include "Momo/Core/SceneFile.h"
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
        // An empty scene, in case the one asked for doesn't load
        m_ActiveScene = Scene();
        m_Camera = Camera::LookingAlong(m_SceneCamera.position, glm::vec3(0.0f, 1.0f, 0.0f), glm::normalize(m_SceneCamera.target - m_SceneCamera.position));
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

    bool Application::LoadScene(const std::filesystem::path &path)
    {
        // A glTF file on its own is a scene of one entity
        SceneDescription description;
        std::filesystem::path savePath;
        if (path.extension() == ".json")
        {
            std::string error;
            std::optional<SceneDescription> read = ReadSceneFile(path, error);
            if (!read) {
                LOG_ERROR("Momo", "Failed to load scene: {}", error);
                return false;
            }
            description = std::move(*read);
            savePath = path;
        }
        else
        {
            if (!std::filesystem::exists(path)) {
                LOG_ERROR("Momo", "Scene file does not exist: {}", path.generic_string());
                return false;
            }
            description.entities.push_back({ .name = path.stem().string(), .asset = path.filename() });
            savePath = path.parent_path() / (path.stem().string() + ".scene.json");
        }
        m_SavePath = savePath;
        m_SavePathInput.fill('\0');

        // Instances game code added (a player character) outlive the scene they were added to
        if (!m_ActiveScene)
            m_ActiveScene = Scene();
        RemoveSavedInstances();

        // The first camera, sun and sky settings any asset brings, for whatever the scene file doesn't set
        std::optional<Assets::CameraSource> assetCamera;
        std::optional<Assets::SunSource> assetSun;
        Assets::SkySource assetSky;
        for (const EntityDescription& entity : description.entities)
        {
            std::optional<EntityId> id = AddEntity(entity, path.parent_path());
            if (!id)
                continue;
            const SceneInstance& instance = *m_ActiveScene->Find(*id);
            const LoadedAsset& asset = m_LoadedAssets.at(instance.source);
            glm::mat3 rotation = glm::mat3(instance.transform);
            if (!assetCamera && asset.camera)
                assetCamera = Assets::CameraSource{ .position = glm::vec3(instance.transform * glm::vec4(asset.camera->position, 1.0f)),
                                                    .forward = glm::normalize(rotation * asset.camera->forward) };
            if (!assetSun && asset.sun)
            {
                assetSun = asset.sun;
                assetSun->towardSun = glm::normalize(rotation * asset.sun->towardSun);
            }
            if (!assetSky.zenith) assetSky.zenith = asset.sky.zenith;
            if (!assetSky.horizon) assetSky.horizon = asset.sky.horizon;
            if (!assetSky.ground) assetSky.ground = asset.sky.ground;
            if (!assetSky.exposure) assetSky.exposure = asset.sky.exposure;
        }
        LOG_INFO("Momo", "Loaded scene {}: {} entities", path.generic_string(), description.entities.size());

        constexpr glm::vec3 up(0.0f, 1.0f, 0.0f);
        if (description.camera && glm::length(description.camera->target - description.camera->position) > 1e-4f)
            m_SceneCamera = *description.camera;
        else if (assetCamera)
            m_SceneCamera = SceneCameraDescription{ .position = assetCamera->position, .target = assetCamera->position + assetCamera->forward };
        else
            m_SceneCamera = SceneCameraDescription{ .position = { 0.0f, 2.0f, 0.0f }, .target = { 0.0f, 2.0f, -1.0f } };
        m_Camera = Camera::LookingAlong(m_SceneCamera.position, up, glm::normalize(m_SceneCamera.target - m_SceneCamera.position));

        Assets::SkySource sky = description.sky;
        if (!sky.zenith) sky.zenith = assetSky.zenith;
        if (!sky.horizon) sky.horizon = assetSky.horizon;
        if (!sky.ground) sky.ground = assetSky.ground;
        if (!sky.exposure) sky.exposure = assetSky.exposure;
        ApplySceneLighting(description.sun ? description.sun : assetSun, sky);

        RefreshLightsAndShadowBounds();
        // Build the ray tracing geometry right away rather than after the usual pause
        MarkStaticGeometryChanged();
        m_StaticRebuildAt = std::chrono::steady_clock::now();
        return true;
    }

    bool Application::OpenScene(const std::filesystem::path& path)
    {
        return LoadScene(path);
    }

    void Application::NewScene()
    {
        RemoveSavedInstances();
        m_SavePath = "Assets/Scenes/Untitled.scene.json";
        m_SavePathInput.fill('\0');
        RefreshLightsAndShadowBounds();
    }

    void Application::RemoveSavedInstances()
    {
        std::vector<EntityId> saved;
        for (const SceneInstance& instance : m_ActiveScene->GetInstances())
            if (instance.saved)
                saved.push_back(instance.id);
        for (EntityId id : saved)
            RemoveInstance(id);
    }

    // Point lights come from the entities' assets, moved with them, and sun
    // shadows cover everything that doesn't move
    void Application::RefreshLightsAndShadowBounds()
    {
        std::vector<Assets::PointLight> pointLights;
        glm::vec3 boundsMin(std::numeric_limits<float>::max());
        glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
        for (const SceneInstance& instance : m_ActiveScene->GetInstances())
        {
            for (Assets::PointLight light : m_AssetRegistry.Get(instance.model).pointLights)
            {
                light.position = glm::vec3(instance.transform * glm::vec4(light.position, 1.0f));
                if (light.name.empty())
                    light.name = instance.name;
                pointLights.push_back(std::move(light));
            }
            if (!instance.isStatic)
                continue;
            Bounds model = GetModelBounds(instance.model);
            if (model.min.x > model.max.x)
                continue;
            for (int corner = 0; corner < 8; ++corner)
            {
                glm::vec3 local(corner & 1 ? model.max.x : model.min.x, corner & 2 ? model.max.y : model.min.y, corner & 4 ? model.max.z : model.min.z);
                glm::vec3 world = glm::vec3(instance.transform * glm::vec4(local, 1.0f));
                boundsMin = glm::min(boundsMin, world);
                boundsMax = glm::max(boundsMax, world);
            }
        }
        if (boundsMin.x <= boundsMax.x)
            m_Renderer.SetShadowBounds(boundsMin, boundsMax);

        // Leave the light editor's changes alone unless the scene's lights themselves changed
        bool sameLights = pointLights.size() == m_SceneLights.size();
        for (size_t i = 0; sameLights && i < pointLights.size(); ++i)
            sameLights = pointLights[i].position == m_SceneLights[i].position && pointLights[i].intensity == m_SceneLights[i].intensity &&
                         pointLights[i].name == m_SceneLights[i].name;
        if (!sameLights || !m_LightsCollected)
        {
            m_SceneLights = pointLights;
            m_LightsCollected = true;
            m_Renderer.SetPointLights(pointLights);
            CollectEditableLights(pointLights);
        }
    }

    Bounds Application::GetModelBounds(Assets::ModelHandle model) const
    {
        Bounds bounds{ glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest()) };
        for (const Assets::MeshHandle& meshHandle : m_AssetRegistry.Get(model).meshes)
        {
            const Bounds& mesh = GetMeshBounds(meshHandle);
            if (mesh.min.x > mesh.max.x)
                continue;
            const glm::mat4& toModel = m_AssetRegistry.Get(meshHandle).localTransform;
            for (int corner = 0; corner < 8; ++corner)
            {
                glm::vec3 local(corner & 1 ? mesh.max.x : mesh.min.x, corner & 2 ? mesh.max.y : mesh.min.y, corner & 4 ? mesh.max.z : mesh.min.z);
                glm::vec3 point = glm::vec3(toModel * glm::vec4(local, 1.0f));
                bounds.min = glm::min(bounds.min, point);
                bounds.max = glm::max(bounds.max, point);
            }
        }
        return bounds;
    }

    const Bounds& Application::GetMeshBounds(Assets::MeshHandle mesh) const
    {
        auto it = m_MeshBounds.find(mesh);
        if (it != m_MeshBounds.end())
            return it->second;
        Bounds bounds{ glm::vec3(std::numeric_limits<float>::max()), glm::vec3(std::numeric_limits<float>::lowest()) };
        for (const Geometry::Vertex& vertex : m_AssetRegistry.Get(mesh).meshData.vertices)
        {
            bounds.min = glm::min(bounds.min, vertex.position);
            bounds.max = glm::max(bounds.max, vertex.position);
        }
        return m_MeshBounds.emplace(mesh, bounds).first->second;
    }

    std::optional<PickHit> Application::PickInstance(glm::vec3 origin, glm::vec3 direction, bool savedOnly) const
    {
        std::optional<PickHit> nearest;
        for (const SceneInstance& instance : m_ActiveScene->GetInstances())
        {
            if (savedOnly && !instance.saved)
                continue;
            for (const Assets::MeshHandle& meshHandle : m_AssetRegistry.Get(instance.model).meshes)
            {
                const Assets::Mesh& mesh = m_AssetRegistry.Get(meshHandle);
                // The ray in the mesh's own space. The direction isn't
                // renormalized, so distances along it stay world distances.
                glm::mat4 toLocal = glm::inverse(instance.transform * mesh.localTransform);
                glm::vec3 o = glm::vec3(toLocal * glm::vec4(origin, 1.0f));
                glm::vec3 d = glm::mat3(toLocal) * direction;

                // Skip the triangles unless the ray crosses the mesh's box (slab test)
                const Bounds& bounds = GetMeshBounds(meshHandle);
                glm::vec3 inverse = 1.0f / d;
                glm::vec3 t0 = (bounds.min - o) * inverse;
                glm::vec3 t1 = (bounds.max - o) * inverse;
                glm::vec3 tNear = glm::min(t0, t1);
                glm::vec3 tFar = glm::max(t0, t1);
                float enter = std::max({ tNear.x, tNear.y, tNear.z, 0.0f });
                float exit = std::min({ tFar.x, tFar.y, tFar.z });
                if (!(enter <= exit) || (nearest && enter > nearest->distance))
                    continue;

                const std::vector<Geometry::Vertex>& vertices = mesh.meshData.vertices;
                const std::vector<uint32_t>& indices = mesh.meshData.indices;
                for (size_t i = 0; i + 2 < indices.size(); i += 3)
                {
                    // Moller-Trumbore, both sides
                    const glm::vec3& v0 = vertices[indices[i]].position;
                    glm::vec3 e1 = vertices[indices[i + 1]].position - v0;
                    glm::vec3 e2 = vertices[indices[i + 2]].position - v0;
                    glm::vec3 p = glm::cross(d, e2);
                    float determinant = glm::dot(e1, p);
                    if (std::abs(determinant) < 1e-12f)
                        continue;
                    float inverseDeterminant = 1.0f / determinant;
                    glm::vec3 s = o - v0;
                    float u = glm::dot(s, p) * inverseDeterminant;
                    if (u < 0.0f || u > 1.0f)
                        continue;
                    glm::vec3 q = glm::cross(s, e1);
                    float v = glm::dot(d, q) * inverseDeterminant;
                    if (v < 0.0f || u + v > 1.0f)
                        continue;
                    float t = glm::dot(e2, q) * inverseDeterminant;
                    if (t > 1e-4f && (!nearest || t < nearest->distance))
                        nearest = PickHit{ .entity = instance.id, .distance = t, .point = origin + direction * t };
                }
            }
        }
        return nearest;
    }

    const Application::LoadedAsset* Application::LoadAsset(const std::filesystem::path& path)
    {
        // Entities using the same file share one copy of it
        std::error_code ignored;
        std::filesystem::path key = std::filesystem::weakly_canonical(std::filesystem::absolute(path), ignored);
        if (auto it = m_LoadedAssets.find(key); it != m_LoadedAssets.end())
            return &it->second;

        if (!std::filesystem::exists(path)) {
            LOG_ERROR("Momo", "Model file does not exist: {}", path.generic_string());
            return nullptr;
        }
        auto source = m_ModelLoader->LoadModel(path);
        if (!source.has_value()) {
            LOG_ERROR("Momo", "Failed to load model from path: {}", path.generic_string());
            return nullptr;
        }
        // The ray tracing build reads the registry on a worker thread, and
        // registering can reallocate it. Models loaded in OnAttach come before
        // the build starts; any loaded later wait for it to finish.
        if (m_PendingRayTracingGeometry.valid())
            m_PendingRayTracingGeometry.wait();
        LoadedAsset asset{ .path = key, .camera = source->camera, .sun = source->sun, .sky = source->sky, .handle = {} };
        asset.handle = m_AssetRegistry.RegisterModel(std::move(*source));
        return &m_LoadedAssets.emplace(key, std::move(asset)).first->second;
    }

    std::optional<Assets::ModelHandle> Application::LoadModel(const std::filesystem::path& path)
    {
        const LoadedAsset* asset = LoadAsset(path);
        if (!asset)
            return std::nullopt;
        return asset->handle;
    }

    EntityId Application::AddDynamicInstance(Assets::ModelHandle model, const glm::mat4& transform)
    {
        return m_ActiveScene->AddInstance({ .model = model, .transform = transform, .isStatic = false, .name = {}, .source = {}, .saved = false });
    }

    std::optional<EntityId> Application::AddEntity(const EntityDescription& entity, const std::filesystem::path& directory)
    {
        const LoadedAsset* asset = LoadAsset(directory / entity.asset); // An absolute asset path stays as it is
        if (!asset)
            return std::nullopt;
        if (entity.isStatic)
            MarkStaticGeometryChanged();
        return m_ActiveScene->AddInstance({ .model = asset->handle, .transform = entity.Transform(), .isStatic = entity.isStatic,
                                            .name = entity.name, .source = asset->path, .saved = true });
    }

    void Application::RemoveInstance(EntityId id)
    {
        const SceneInstance* instance = m_ActiveScene->Find(id);
        if (!instance)
            return;
        if (instance->isStatic)
            MarkStaticGeometryChanged();
        m_ActiveScene->Remove(id);
    }

    void Application::SetInstanceTransform(EntityId id, const glm::mat4& transform)
    {
        SceneInstance* instance = m_ActiveScene->Find(id);
        if (!instance)
            return;
        instance->transform = transform;
        if (instance->isStatic)
            MarkStaticGeometryChanged();
    }

    void Application::SetInstanceStatic(EntityId id, bool isStatic)
    {
        SceneInstance* instance = m_ActiveScene->Find(id);
        if (!instance || instance->isStatic == isStatic)
            return;
        instance->isStatic = isStatic;
        MarkStaticGeometryChanged();
    }

    void Application::SetInstanceName(EntityId id, const std::string& name)
    {
        if (SceneInstance* instance = m_ActiveScene->Find(id))
            instance->name = name;
    }

    std::vector<EntityId> Application::ReplaceEntities(const std::vector<EntityDescription>& entities)
    {
        RemoveSavedInstances();
        std::vector<EntityId> ids;
        for (const EntityDescription& entity : entities)
            ids.push_back(AddEntity(entity).value_or(0));
        return ids;
    }

    // Static geometry is rebuilt a moment after the last change, so dragging
    // something around doesn't start a build per frame
    void Application::MarkStaticGeometryChanged()
    {
        m_StaticGeometryChanged = true;
        m_StaticRebuildAt = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
    }

    void Application::RebuildStaticGeometryIfChanged()
    {
        if (!m_StaticGeometryChanged || m_PendingRayTracingGeometry.valid() || std::chrono::steady_clock::now() < m_StaticRebuildAt)
            return;
        m_StaticGeometryChanged = false;
        RefreshLightsAndShadowBounds();
        bool anyStatic = std::any_of(m_ActiveScene->GetInstances().begin(), m_ActiveScene->GetInstances().end(),
                                     [](const SceneInstance& instance) { return instance.isStatic; });
        if (!anyStatic)
        {
            // Nothing to build a BVH over. The ray traced view keeps the last
            // scene it was given, but there is nothing left to collide with.
            m_Collision.reset();
            return;
        }
        StartRayTracingGeometryBuild();
    }

    SceneDescription Application::DescribeScene() const
    {
        SceneDescription description;
        description.camera = m_SceneCamera;
        description.sun = Assets::SunSource{ .towardSun = m_Lighting.SunDirection(), .color = m_Lighting.sunColor, .intensity = m_Lighting.sunIntensity };
        description.sky = Assets::SkySource{ .zenith = m_Lighting.skyZenith, .horizon = m_Lighting.skyHorizon,
                                             .ground = m_Lighting.skyGround, .exposure = m_Lighting.exposure };
        for (const SceneInstance& instance : m_ActiveScene->GetInstances())
        {
            if (!instance.saved)
                continue;
            EntityDescription entity;
            entity.name = instance.name;
            entity.asset = instance.source;
            entity.isStatic = instance.isStatic;
            DecomposeTransform(instance.transform, entity.position, entity.rotation, entity.scale);
            description.entities.push_back(std::move(entity));
        }
        return description;
    }

    bool Application::SaveScene(const std::filesystem::path& path, std::string& error)
    {
        if (!m_ActiveScene) {
            error = "No scene is loaded";
            return false;
        }
        SceneDescription description = DescribeScene();
        std::error_code ignored;
        std::filesystem::path directory = std::filesystem::weakly_canonical(std::filesystem::absolute(path), ignored).parent_path();
        auto topFolder = [](const std::filesystem::path& p) {
            std::filesystem::path below = p.relative_path();
            return below.empty() ? std::filesystem::path() : *below.begin();
        };
        for (EntityDescription& entity : description.entities)
        {
            // Relative to the scene file, unless the two only share the root
            // (/home/... from /tmp/...), where an absolute path reads better
            std::filesystem::path relative = entity.asset.lexically_relative(directory);
            bool sharesMoreThanRoot = entity.asset.root_path() == directory.root_path() &&
                                      !topFolder(directory).empty() && topFolder(entity.asset) == topFolder(directory);
            if (!relative.empty() && sharesMoreThanRoot)
                entity.asset = relative;
        }
        if (!WriteSceneFile(description, path, error))
            return false;
        m_SavePath = path;
        m_SavePathInput.fill('\0');
        LOG_INFO("Momo", "Saved scene to {}", path.generic_string());
        return true;
    }

    void Application::SetEditing(bool editing)
    {
        m_Editing = editing;
        m_Window->SetCursorCaptured(!editing && !m_ShowSettings);
    }

    void Application::SetCursorCaptured(bool captured)
    {
        m_Window->SetCursorCaptured(captured);
    }

    std::optional<RaycastHit> Application::Raycast(glm::vec3 origin, glm::vec3 direction, float maxDistance) const
    {
        if (!m_Collision)
            return std::nullopt;
        RayTracing::Ray ray{ .origin = origin, .direction = direction };
        std::optional<RayTracing::Hit> hit = RayTracing::IntersectWideBvh(ray, m_Collision->bvh, m_Collision->triangles, 0.0f, maxDistance);
        if (!hit)
            return std::nullopt;
        const RayTracing::TrianglePositions& triangle = m_Collision->triangles[hit->triangle];
        glm::vec3 normal = glm::normalize(glm::cross(glm::vec3(triangle.v1 - triangle.v0), glm::vec3(triangle.v2 - triangle.v0)));
        if (glm::dot(normal, direction) > 0.0f)
            normal = -normal;
        return RaycastHit{ .distance = hit->t, .point = origin + direction * hit->t, .normal = normal };
    }

    // A scene without its own sun or sky keeps the defaults (a sunset)
    void Application::ApplySceneLighting(const std::optional<Assets::SunSource>& sun, const Assets::SkySource& sky)
    {
        Renderer::SceneLighting lighting{};
        if (sun)
        {
            // Inverse of SceneLighting::SunDirection()
            glm::vec3 toward = sun->towardSun;
            lighting.sunElevation = glm::degrees(std::asin(std::clamp(toward.y, -1.0f, 1.0f)));
            lighting.sunAzimuth = glm::degrees(std::atan2(toward.z, toward.x));
            lighting.sunColor = sun->color;
            lighting.sunIntensity = sun->intensity;
        }
        lighting.skyZenith = sky.zenith.value_or(lighting.skyZenith);
        lighting.skyHorizon = sky.horizon.value_or(lighting.skyHorizon);
        lighting.skyGround = sky.ground.value_or(lighting.skyGround);
        lighting.exposure = sky.exposure.value_or(lighting.exposure);
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
        m_Collision = CollisionGeometry{ .bvh = geometry.bvh, .triangles = geometry.positions };

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
            // Typing into a text field isn't walking, quitting or switching modes
            if (!m_Benchmark && ImGui::GetIO().WantTextInput)
                m_InputState.keys = {};
            // While editing, Escape belongs to the editor (deselect, cancel)
            if (m_InputState.keys.IsKeyPressed(Input::Key::Escape) && !m_Editing)
            {
                LOG_INFO("Momo", "Quit requested. Exiting main loop.");
                m_Running = false;
                break;
            }

            UploadRayTracingGeometryIfReady();
            // Started here rather than when the scene loads so layers'
            // OnAttach can still load models into the registry first
            RebuildStaticGeometryIfChanged();

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
                if (!m_Editing)
                    m_Window->SetCursorCaptured(!m_ShowSettings);
            }
            m_ToggleSettingsHeld = toggleSettings;

            if (!m_Benchmark)
            {
                // A frame is built even while the panel is hidden, so stale draw data never lingers
                ImGui_ImplVulkan_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();
                for (auto& layer : m_Layers)
                    layer->OnImGui();
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
                if (!m_Editing || layer->UpdatesWhileEditing())
                    layer->OnUpdate(dt);

            // Benchmarks place the camera themselves and ignore input; screenshots keep the scene's camera
            // While editing, the editor moves the camera
            if (!m_Benchmark && !m_Spec.Screenshot && m_FreeCamera && !m_Editing)
                m_Camera->OnUpdate(dt, m_InputState);
            else if (m_Benchmark && !m_Benchmark->IsFinished())
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

    void Application::CollectEditableLights(const std::vector<Assets::PointLight>& pointLights)
    {
        m_PointLights.clear();
        for (const Assets::PointLight& light : pointLights)
        {
            EditablePointLight editable{ .name = light.name, .position = light.position };
            SplitColor(light.intensity, editable.color, editable.intensity);
            m_PointLights.push_back(editable);
        }
        m_LoadedPointLights = m_PointLights;

        m_Glows.clear();
        for (const auto& [path, asset] : m_LoadedAssets)
        {
            for (const Assets::MeshHandle& meshHandle : m_AssetRegistry.Get(asset.handle).meshes)
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

    void Application::DrawSceneSection()
    {
        if (!ImGui::CollapsingHeader("Scene"))
            return;
        if (m_SavePathInput[0] == '\0')
        {
            std::string initial = m_SavePath.generic_string();
            initial.copy(m_SavePathInput.data(), m_SavePathInput.size() - 1);
        }
        ImGui::InputText("File", m_SavePathInput.data(), m_SavePathInput.size());
        if (ImGui::Button("Save scene"))
        {
            std::string error;
            std::string path = m_SavePathInput.data(); // SaveScene resets the input
            m_SaveStatus = SaveScene(path, error) ? "Saved" : error;
        }
        if (ImGui::Button("Set scene camera from view"))
            SetSceneCamera(m_Camera->GetPosition(), m_Camera->GetPosition() + m_Camera->GetFront());
        ImGui::SetItemTooltip("Where the view starts when the scene is loaded");
        if (!m_SaveStatus.empty())
        {
            ImGui::SameLine();
            ImGui::TextWrapped("%s", m_SaveStatus.c_str());
        }
        ImGui::TextDisabled("Relative paths are from the build directory, whose Assets\nare overwritten from the source tree on every build");
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

        DrawSceneSection();

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
        for (const SceneInstance& instance : m_ActiveScene->GetInstances())
        {
            for (const auto& meshHandle : m_AssetRegistry.Get(instance.model).meshes)
            {
                const Assets::Mesh& mesh = m_AssetRegistry.Get(meshHandle);
                const Renderer::VulkanMaterialData& materialData = m_MaterialCache.GetOrCreate(mesh.materialHandle);

                vulkanModelData.meshes.push_back(Renderer::VulkanMeshData {
                    .gpuMesh = m_MeshCache.GetOrCreate(meshHandle),
                    .localTransform = instance.transform * mesh.localTransform,
                    .materialData = materialData
                });
            }
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
