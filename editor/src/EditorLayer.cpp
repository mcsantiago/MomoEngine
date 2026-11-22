#include "MomoEditor/EditorLayer.h"

#include <Momo/Logging/Logger.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <ImGuizmo.h>

namespace Momo::Editor
{
    namespace
    {
        constexpr float FlySpeed = 4.0f;      // Meters per second
        constexpr float FastFlySpeed = 14.0f; // With Shift
        constexpr float MoveSnap = 0.25f;     // Meters
        constexpr float RotateSnap = 15.0f;   // Degrees
        constexpr float ScaleSnap = 0.1f;
        constexpr size_t UndoLimit = 200;
        constexpr ImU32 SelectionColor = IM_COL32(255, 170, 40, 255);

        bool Same(const EntityDescription& a, const EntityDescription& b)
        {
            return a.name == b.name && a.asset == b.asset && a.position == b.position && a.rotation == b.rotation &&
                   a.scale == b.scale && a.isStatic == b.isStatic;
        }

        bool Same(const std::vector<EntityDescription>& a, const std::vector<EntityDescription>& b)
        {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const auto& x, const auto& y) { return Same(x, y); });
        }

        template <size_t N>
        void SetText(std::array<char, N>& buffer, const std::string& text)
        {
            buffer.fill('\0');
            text.copy(buffer.data(), N - 1);
        }

        bool IsSceneFile(const std::filesystem::path& path)
        {
            return path.filename().string().ends_with(".scene.json");
        }

        bool IsModelFile(const std::filesystem::path& path)
        {
            return path.extension() == ".glb" || path.extension() == ".gltf";
        }
    } // namespace

    void EditorLayer::OnAttach()
    {
        ScanAssets();
        SyncFromApplication();
        if (m_Spec.StartEditing)
            SetEditing(true);
        LOG_INFO("Editor", "F1 switches between editing and playing");
    }

    void EditorLayer::OnImGui()
    {
        if (ImGui::IsKeyPressed(ImGuiKey_F1, false))
            SetEditing(!App().IsEditing());
        if (!App().IsEditing())
            return;

        ImGuizmo::BeginFrame();
        UpdateCamera();
        HandleShortcuts();
        DrawMenuBar();
        DrawEntitiesPanel();
        DrawInspector();
        DrawAssetsPanel();
        DrawFileDialogs();
        DrawSelectionOutline();
        DrawMarkers();
        DrawGizmo();
        HandleViewportClick();
    }

    // --- Editing and playing ----------------------------------------------

    void EditorLayer::SetEditing(bool editing)
    {
        Camera& camera = App().GetCamera();
        if (editing)
        {
            App().SetEditing(true);
            if (m_PlaySnapshot)
            {
                // Back from playing: drop whatever the game did to the scene.
                // (Left alone if nothing changed, to save rebuilding it.)
                SyncFromApplication();
                if (!Same(*m_PlaySnapshot, TakeSnapshot()))
                    ApplySnapshot(*m_PlaySnapshot);
                camera.LookAt(m_EditCameraPosition, m_EditCameraPosition + m_EditCameraFront);
                m_PlaySnapshot.reset();
            }
            else
                SyncFromApplication();
            m_Status = "Editing";
        }
        else
        {
            EndEdit();
            m_Flying = false;
            m_Placing.reset();
            m_PlaySnapshot = TakeSnapshot();
            m_EditCameraPosition = camera.GetPosition();
            m_EditCameraFront = camera.GetFront();
            App().SetEditing(false);
        }
    }

    void EditorLayer::SyncFromApplication()
    {
        SceneDescription description = App().DescribeScene();
        m_Entities.clear();
        size_t next = 0;
        for (const SceneInstance& instance : App().GetScene().GetInstances())
            if (instance.saved)
                m_Entities.push_back({ instance.id, description.entities[next++] });
        if (m_Selected && *m_Selected >= m_Entities.size())
            m_Selected.reset();
    }

    EditorLayer::Snapshot EditorLayer::TakeSnapshot() const
    {
        Snapshot snapshot;
        for (const Entity& entity : m_Entities)
            snapshot.push_back(entity.description);
        return snapshot;
    }

    void EditorLayer::ApplySnapshot(const Snapshot& snapshot)
    {
        std::vector<EntityId> ids = App().ReplaceEntities(snapshot);
        m_Entities.clear();
        for (size_t i = 0; i < snapshot.size(); ++i)
            if (ids[i] != 0)
                m_Entities.push_back({ ids[i], snapshot[i] });
        if (m_Selected && *m_Selected >= m_Entities.size())
            m_Selected.reset();
    }

    void EditorLayer::PushToApplication(const Entity& entity)
    {
        App().SetInstanceTransform(entity.id, entity.description.Transform());
        App().SetInstanceStatic(entity.id, entity.description.isStatic);
        App().SetInstanceName(entity.id, entity.description.name);
    }

    // --- Undo ---------------------------------------------------------------

    void EditorLayer::RecordUndo()
    {
        m_UndoStack.push_back(TakeSnapshot());
        if (m_UndoStack.size() > UndoLimit)
            m_UndoStack.erase(m_UndoStack.begin());
        m_RedoStack.clear();
        m_Modified = true;
    }

    void EditorLayer::BeginEdit()
    {
        if (!m_EditStart)
            m_EditStart = TakeSnapshot();
    }

    void EditorLayer::EndEdit()
    {
        if (!m_EditStart)
            return;
        Snapshot before = std::move(*m_EditStart);
        m_EditStart.reset();
        if (Same(before, TakeSnapshot()))
            return;
        m_UndoStack.push_back(std::move(before));
        if (m_UndoStack.size() > UndoLimit)
            m_UndoStack.erase(m_UndoStack.begin());
        m_RedoStack.clear();
        m_Modified = true;
    }

    void EditorLayer::Undo()
    {
        EndEdit();
        if (m_UndoStack.empty())
            return;
        m_RedoStack.push_back(TakeSnapshot());
        ApplySnapshot(m_UndoStack.back());
        m_UndoStack.pop_back();
        m_Modified = true;
        m_Status = "Undid";
    }

    void EditorLayer::Redo()
    {
        EndEdit();
        if (m_RedoStack.empty())
            return;
        m_UndoStack.push_back(TakeSnapshot());
        ApplySnapshot(m_RedoStack.back());
        m_RedoStack.pop_back();
        m_Modified = true;
        m_Status = "Redid";
    }

    // --- Entities -----------------------------------------------------------

    void EditorLayer::AddEntity(const std::filesystem::path& asset, glm::vec3 position)
    {
        EntityDescription description;
        description.name = asset.stem().string();
        description.asset = std::filesystem::absolute(asset);
        description.position = position;
        RecordUndo();
        std::optional<EntityId> id = App().AddEntity(description);
        if (!id)
        {
            m_UndoStack.pop_back();
            m_Status = "Couldn't load " + asset.generic_string();
            return;
        }
        // The application's copy has the asset path in its canonical form
        description.asset = App().GetScene().Find(*id)->source;
        m_Entities.push_back({ *id, description });
        m_Selected = m_Entities.size() - 1;
        m_Status = "Added " + description.name;
    }

    void EditorLayer::AddMarker()
    {
        // On the ground ahead of the camera, facing the way it looks
        const Camera& camera = App().GetCamera();
        glm::vec3 forward = camera.GetFront();
        forward.y = 0.0f;
        forward = glm::length(forward) > 1e-3f ? glm::normalize(forward) : glm::vec3(0.0f, 0.0f, -1.0f);
        EntityDescription description;
        description.name = "Marker";
        description.position = camera.GetPosition() + forward * 4.0f;
        description.position.y = 0.0f;
        description.rotation.y = glm::degrees(std::atan2(forward.x, forward.z));
        description.isStatic = false;
        RecordUndo();
        std::optional<EntityId> id = App().AddEntity(description);
        if (!id)
            return;
        m_Entities.push_back({ *id, description });
        m_Selected = m_Entities.size() - 1;
        m_Status = "Added a marker";
    }

    void EditorLayer::DuplicateSelected()
    {
        Entity* selected = Selected();
        if (!selected)
            return;
        EntityDescription copy = selected->description;
        // Beside the original rather than exactly inside it
        Bounds bounds = App().GetModelBounds(App().GetScene().Find(selected->id)->model);
        float width = bounds.min.x <= bounds.max.x ? (bounds.max.x - bounds.min.x) * std::abs(copy.scale.x) : 1.0f;
        glm::vec3 side = glm::normalize(glm::vec3(copy.Transform()[0])); // Its own X axis
        copy.position += side * std::clamp(width, 0.2f, 5.0f);
        RecordUndo();
        std::optional<EntityId> id = App().AddEntity(copy);
        if (!id)
            return;
        m_Entities.push_back({ *id, copy });
        m_Selected = m_Entities.size() - 1;
        m_Status = "Duplicated " + copy.name;
    }

    void EditorLayer::DeleteSelected()
    {
        Entity* selected = Selected();
        if (!selected)
            return;
        RecordUndo();
        m_Status = "Deleted " + selected->description.name;
        App().RemoveInstance(selected->id);
        m_Entities.erase(m_Entities.begin() + static_cast<std::ptrdiff_t>(*m_Selected));
        m_Selected.reset();
    }

    void EditorLayer::FocusSelected()
    {
        Entity* selected = Selected();
        if (!selected)
            return;
        const SceneInstance* instance = App().GetScene().Find(selected->id);
        Bounds bounds = App().GetModelBounds(instance->model);
        glm::vec3 center = selected->description.position;
        float radius = 1.0f;
        if (bounds.min.x <= bounds.max.x)
        {
            center = glm::vec3(instance->transform * glm::vec4((bounds.min + bounds.max) * 0.5f, 1.0f));
            glm::vec3 corner = glm::vec3(instance->transform * glm::vec4(bounds.max, 1.0f));
            radius = std::max(glm::length(corner - center), 0.1f);
        }
        // Far enough back to fit it in the 45 degree view, but inside the far plane
        Camera& camera = App().GetCamera();
        float distance = std::min(radius * 2.8f + 0.3f, 80.0f);
        camera.LookAt(center - camera.GetFront() * distance, center);
    }

    // --- Files --------------------------------------------------------------

    void EditorLayer::NewScene()
    {
        App().NewScene();
        SyncFromApplication();
        m_Selected.reset();
        m_UndoStack.clear();
        m_RedoStack.clear();
        m_PlaySnapshot.reset();
        m_Modified = false;
        m_Status = "New scene";
    }

    void EditorLayer::OpenScene(const std::filesystem::path& path)
    {
        if (!App().OpenScene(path))
        {
            m_Status = "Couldn't open " + path.generic_string() + " (see the log)";
            return;
        }
        m_Selected.reset();
        m_UndoStack.clear();
        m_RedoStack.clear();
        m_PlaySnapshot.reset();
        SyncFromApplication();
        m_Modified = false;
        m_Status = "Opened " + path.generic_string();
    }

    void EditorLayer::Save()
    {
        SaveAs(App().GetScenePath());
    }

    void EditorLayer::SaveAs(const std::filesystem::path& path)
    {
        EndEdit();
        std::string error;
        if (App().SaveScene(path, error))
        {
            m_Modified = false;
            m_Status = "Saved " + path.generic_string();
            ScanAssets();
        }
        else
            m_Status = error;
    }

    void EditorLayer::ScanAssets()
    {
        m_Assets.clear();
        m_Scenes.clear();
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator it(m_Spec.AssetRoot, std::filesystem::directory_options::skip_permission_denied, error), end;
             !error && it != end; it.increment(error))
        {
            if (!it->is_regular_file(error))
                continue;
            std::filesystem::path relative = it->path().lexically_relative(m_Spec.AssetRoot);
            if (IsModelFile(relative))
                m_Assets.push_back(relative);
            else if (IsSceneFile(relative))
                m_Scenes.push_back(relative);
        }
        std::sort(m_Assets.begin(), m_Assets.end());
        std::sort(m_Scenes.begin(), m_Scenes.end());
    }

    // --- Input --------------------------------------------------------------

    void EditorLayer::HandleShortcuts()
    {
        ImGuiIO& io = ImGui::GetIO();
        if (io.WantTextInput)
            return;
        auto pressed = [](ImGuiKey key) { return ImGui::IsKeyPressed(key, false); };

        if (io.KeyCtrl)
        {
            if (pressed(ImGuiKey_Z))
                io.KeyShift ? Redo() : Undo();
            if (pressed(ImGuiKey_Y))
                Redo();
            if (pressed(ImGuiKey_D))
                DuplicateSelected();
            if (pressed(ImGuiKey_S))
                Save();
            return;
        }
        if (pressed(ImGuiKey_Escape))
        {
            if (m_Placing)
                m_Placing.reset();
            else
                m_Selected.reset();
        }
        if (pressed(ImGuiKey_Delete))
            DeleteSelected();
        // While flying, W and E move the camera
        if (m_Flying)
            return;
        if (pressed(ImGuiKey_W)) m_GizmoMode = GizmoMode::Move;
        if (pressed(ImGuiKey_E)) m_GizmoMode = GizmoMode::Rotate;
        if (pressed(ImGuiKey_R)) m_GizmoMode = GizmoMode::Scale;
        if (pressed(ImGuiKey_F)) FocusSelected();
    }

    void EditorLayer::UpdateCamera()
    {
        ImGuiIO& io = ImGui::GetIO();
        Camera& camera = App().GetCamera();
        if (!m_Flying && ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.WantCaptureMouse)
        {
            m_Flying = true;
            App().SetCursorCaptured(true);
        }
        else if (m_Flying && !ImGui::IsMouseDown(ImGuiMouseButton_Right))
        {
            m_Flying = false;
            App().SetCursorCaptured(false);
        }

        // The camera is updated every frame so it keeps track of the mouse,
        // and doesn't jump when flying starts; it only turns while the cursor
        // is captured, and only gets the keys while flying
        Input::InputState input = App().GetInput();
        if (!m_Flying || io.WantTextInput)
            input.keys = {};
        camera.SetMovementSpeed(io.KeyShift ? FastFlySpeed : FlySpeed);
        camera.OnUpdate(io.DeltaTime, input);
        if (m_Flying && !io.WantTextInput)
        {
            float rise = (ImGui::IsKeyDown(ImGuiKey_E) ? 1.0f : 0.0f) - (ImGui::IsKeyDown(ImGuiKey_Q) ? 1.0f : 0.0f);
            if (rise != 0.0f)
            {
                glm::vec3 position = camera.GetPosition() + glm::vec3(0.0f, rise * (io.KeyShift ? FastFlySpeed : FlySpeed) * io.DeltaTime, 0.0f);
                camera.LookAt(position, position + camera.GetFront());
            }
        }
    }

    void EditorLayer::MouseRay(glm::vec3& origin, glm::vec3& direction) const
    {
        // The application's projection has y down, like the screen
        ImGuiIO& io = ImGui::GetIO();
        const Camera& camera = App().GetCamera();
        glm::vec2 ndc(io.MousePos.x / io.DisplaySize.x * 2.0f - 1.0f, io.MousePos.y / io.DisplaySize.y * 2.0f - 1.0f);
        glm::mat4 toWorld = glm::inverse(App().GetProjectionMatrix() * camera.GetViewMatrix());
        glm::vec4 farPoint = toWorld * glm::vec4(ndc, 1.0f, 1.0f);
        origin = camera.GetPosition();
        direction = glm::normalize(glm::vec3(farPoint) / farPoint.w - origin);
    }

    void EditorLayer::HandleViewportClick()
    {
        ImGuiIO& io = ImGui::GetIO();
        if (m_Flying || io.WantCaptureMouse || !ImGui::IsMouseClicked(ImGuiMouseButton_Left))
            return;
        // A click on the gizmo is the start of a drag, not a new selection
        if (!m_Placing && Selected() && (ImGuizmo::IsOver() || ImGuizmo::IsUsing()))
            return;

        // Markers are drawn over everything, so they're picked before it too
        if (!m_Placing)
        {
            if (std::optional<size_t> marker = MarkerUnderMouse())
            {
                m_Selected = marker;
                return;
            }
        }

        glm::vec3 origin, direction;
        MouseRay(origin, direction);
        std::optional<PickHit> hit = App().PickInstance(origin, direction, true);
        if (m_Placing)
        {
            // On whatever is under the mouse, else on the ground plane, else just ahead
            glm::vec3 position = origin + direction * 5.0f;
            if (hit)
                position = hit->point;
            else if (direction.y < -1e-4f)
                position = origin + direction * (-origin.y / direction.y);
            AddEntity(m_Spec.AssetRoot / *m_Placing, position);
            if (!io.KeyShift) // Shift keeps placing more of the same
                m_Placing.reset();
            return;
        }

        m_Selected.reset();
        if (hit)
            for (size_t i = 0; i < m_Entities.size(); ++i)
                if (m_Entities[i].id == hit->entity)
                    m_Selected = i;
    }

    // --- Drawn in the scene -------------------------------------------------

    void EditorLayer::DrawGizmo()
    {
        Entity* selected = Selected();
        if (!selected)
        {
            if (m_GizmoInUse)
                EndEdit();
            m_GizmoInUse = false;
            return;
        }

        ImGuiIO& io = ImGui::GetIO();
        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist(ImGui::GetBackgroundDrawList()); // Behind the panels
        ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);
        ImGuizmo::Enable(!m_Flying);
        ImGuizmo::AllowAxisFlip(false); // Up stays up, whichever side the camera is on

        glm::mat4 view = App().GetCamera().GetViewMatrix();
        glm::mat4 projection = App().GetProjectionMatrix();
        projection[1][1] *= -1.0f; // ImGuizmo expects y up
        glm::mat4 transform = selected->description.Transform();

        ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
        ImGuizmo::MODE mode = ImGuizmo::WORLD;
        float snap[3] = { MoveSnap, MoveSnap, MoveSnap };
        if (m_GizmoMode == GizmoMode::Rotate)
        {
            operation = ImGuizmo::ROTATE;
            mode = ImGuizmo::LOCAL;
            snap[0] = snap[1] = snap[2] = RotateSnap;
        }
        else if (m_GizmoMode == GizmoMode::Scale)
        {
            operation = ImGuizmo::SCALE;
            mode = ImGuizmo::LOCAL;
            snap[0] = snap[1] = snap[2] = ScaleSnap;
        }
        bool changed = ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), operation, mode,
                                            glm::value_ptr(transform), nullptr, io.KeyCtrl ? snap : nullptr);

        bool inUse = ImGuizmo::IsUsing();
        if (inUse && !m_GizmoInUse)
            BeginEdit();
        if (changed)
        {
            // Only what this gizmo edits is taken from the matrix, so moving
            // something doesn't nudge its rotation or scale by rounding
            glm::vec3 position, rotation, scale;
            DecomposeTransform(transform, position, rotation, scale);
            if (m_GizmoMode == GizmoMode::Move) selected->description.position = position;
            if (m_GizmoMode == GizmoMode::Rotate) selected->description.rotation = rotation;
            if (m_GizmoMode == GizmoMode::Scale) selected->description.scale = scale;
            PushToApplication(*selected);
        }
        if (!inUse && m_GizmoInUse)
            EndEdit();
        m_GizmoInUse = inUse;
    }

    namespace
    {
        constexpr float MarkerSize = 0.35f;      // Meters: the cross's arms, and half the arrow
        constexpr float MarkerPickRadius = 14.0f; // Pixels around its origin

        // A world position on screen; nothing if it's behind the camera
        std::optional<ImVec2> ToScreen(const glm::mat4& viewProjection, glm::vec3 world)
        {
            glm::vec4 clip = viewProjection * glm::vec4(world, 1.0f);
            if (clip.w < 0.05f)
                return std::nullopt;
            ImVec2 size = ImGui::GetIO().DisplaySize;
            return ImVec2((clip.x / clip.w * 0.5f + 0.5f) * size.x, (clip.y / clip.w * 0.5f + 0.5f) * size.y);
        }
    }

    // Markers have no model, so the editor draws them: a cross where they
    // are, an arrow along the way they face (their +Z), and their name
    void EditorLayer::DrawMarkers()
    {
        glm::mat4 viewProjection = App().GetProjectionMatrix() * App().GetCamera().GetViewMatrix();
        ImDrawList* draw = ImGui::GetBackgroundDrawList();
        for (size_t i = 0; i < m_Entities.size(); ++i)
        {
            const EntityDescription& description = m_Entities[i].description;
            if (description.asset)
                continue;
            glm::mat4 transform = ComposeTransform(description.position, description.rotation, glm::vec3(1.0f));
            auto at = [&](glm::vec3 local) { return ToScreen(viewProjection, glm::vec3(transform * glm::vec4(local, 1.0f))); };
            ImU32 color = m_Selected == i ? IM_COL32(255, 170, 40, 255) : IM_COL32(90, 200, 255, 255);
            auto line = [&](glm::vec3 a, glm::vec3 b) {
                std::optional<ImVec2> from = at(a), to = at(b);
                if (from && to)
                    draw->AddLine(*from, *to, color, 2.0f);
            };
            const float s = MarkerSize;
            line({ -s, 0, 0 }, { s, 0, 0 });
            line({ 0, 0, -s }, { 0, 0, s * 2 });
            line({ 0, 0, 0 }, { 0, s * 2, 0 });
            line({ 0, 0, s * 2 }, { -s * 0.4f, 0, s * 1.4f });
            line({ 0, 0, s * 2 }, { s * 0.4f, 0, s * 1.4f });
            if (std::optional<ImVec2> label = at({ 0, s * 2, 0 }))
                draw->AddText(ImVec2(label->x + 6.0f, label->y - 8.0f), color, description.name.c_str());
        }
    }

    std::optional<size_t> EditorLayer::MarkerUnderMouse() const
    {
        glm::mat4 viewProjection = App().GetProjectionMatrix() * App().GetCamera().GetViewMatrix();
        ImVec2 mouse = ImGui::GetIO().MousePos;
        std::optional<size_t> nearest;
        float nearestDistance = MarkerPickRadius;
        for (size_t i = 0; i < m_Entities.size(); ++i)
        {
            if (m_Entities[i].description.asset)
                continue;
            std::optional<ImVec2> screen = ToScreen(viewProjection, m_Entities[i].description.position);
            if (!screen)
                continue;
            float distance = std::hypot(screen->x - mouse.x, screen->y - mouse.y);
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearest = i;
            }
        }
        return nearest;
    }

    void EditorLayer::DrawSelectionOutline()
    {
        Entity* selected = Selected();
        if (!selected)
            return;
        const SceneInstance* instance = App().GetScene().Find(selected->id);
        Bounds bounds = App().GetModelBounds(instance->model);
        if (bounds.min.x > bounds.max.x)
            return;

        // The model's box, moved with it, in clip space
        ImGuiIO& io = ImGui::GetIO();
        glm::mat4 toClip = App().GetProjectionMatrix() * App().GetCamera().GetViewMatrix() * instance->transform;
        glm::vec4 corners[8];
        for (int i = 0; i < 8; ++i)
            corners[i] = toClip * glm::vec4(i & 1 ? bounds.max.x : bounds.min.x, i & 2 ? bounds.max.y : bounds.min.y, i & 4 ? bounds.max.z : bounds.min.z, 1.0f);

        ImDrawList* draw = ImGui::GetBackgroundDrawList();
        auto toScreen = [&](const glm::vec4& clip) {
            return ImVec2((clip.x / clip.w * 0.5f + 0.5f) * io.DisplaySize.x, (clip.y / clip.w * 0.5f + 0.5f) * io.DisplaySize.y);
        };
        constexpr float nearW = 0.05f; // Edges are cut where they pass behind the camera
        for (int a = 0; a < 8; ++a)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                int b = a | (1 << axis);
                if (b == a)
                    continue;
                glm::vec4 p = corners[a], q = corners[b];
                if (p.w < nearW && q.w < nearW)
                    continue;
                if (p.w < nearW) p = glm::mix(q, p, (q.w - nearW) / (q.w - p.w));
                if (q.w < nearW) q = glm::mix(p, q, (p.w - nearW) / (p.w - q.w));
                draw->AddLine(toScreen(p), toScreen(q), SelectionColor, 1.5f);
            }
        }
    }

    // --- Panels -------------------------------------------------------------

    void EditorLayer::DrawMenuBar()
    {
        if (!ImGui::BeginMainMenuBar())
            return;
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("New"))
                NewScene();
            if (ImGui::MenuItem("Open..."))
            {
                m_FileDialog = FileDialog::Open;
                SetText(m_FilePath, App().GetScenePath().generic_string());
                ScanAssets();
            }
            if (ImGui::MenuItem("Save", "Ctrl+S"))
                Save();
            if (ImGui::MenuItem("Save as..."))
            {
                m_FileDialog = FileDialog::SaveAs;
                SetText(m_FilePath, App().GetScenePath().generic_string());
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Quit"))
                App().Quit();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Edit"))
        {
            if (ImGui::MenuItem("Undo", "Ctrl+Z", false, !m_UndoStack.empty()))
                Undo();
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, !m_RedoStack.empty()))
                Redo();
            ImGui::Separator();
            if (ImGui::MenuItem("Add marker"))
                AddMarker();
            ImGui::SetItemTooltip("A named place with no model, for the game to look up (a spawn point, say)");
            if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, Selected() != nullptr))
                DuplicateSelected();
            if (ImGui::MenuItem("Delete", "Del", false, Selected() != nullptr))
                DeleteSelected();
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View"))
        {
            if (ImGui::MenuItem("Look at selection", "F", false, Selected() != nullptr))
                FocusSelected();
            bool settings = App().IsSettingsVisible();
            if (ImGui::MenuItem("Render settings", "Tab", &settings))
                App().SetSettingsVisible(settings);
            ImGui::Separator();
            Camera& camera = App().GetCamera();
            if (ImGui::MenuItem("Set scene camera from view"))
            {
                App().SetSceneCamera(camera.GetPosition(), camera.GetPosition() + camera.GetFront());
                m_Modified = true;
                m_Status = "Scene camera set";
            }
            if (ImGui::MenuItem("Go to scene camera"))
                camera.LookAt(App().GetSceneCamera().position, App().GetSceneCamera().target);
            ImGui::EndMenu();
        }

        ImGui::Separator();
        if (ImGui::RadioButton("Move (W)", m_GizmoMode == GizmoMode::Move)) m_GizmoMode = GizmoMode::Move;
        if (ImGui::RadioButton("Rotate (E)", m_GizmoMode == GizmoMode::Rotate)) m_GizmoMode = GizmoMode::Rotate;
        if (ImGui::RadioButton("Scale (R)", m_GizmoMode == GizmoMode::Scale)) m_GizmoMode = GizmoMode::Scale;
        ImGui::Separator();
        if (ImGui::Button("Play (F1)"))
            SetEditing(false);
        ImGui::Separator();
        ImGui::TextUnformatted((App().GetScenePath().filename().string() + (m_Modified ? " *" : "")).c_str());
        ImGui::Separator();
        if (m_Placing)
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "Click in the scene to place %s (Shift for more, Esc to cancel)", m_Placing->filename().string().c_str());
        else if (!App().IsStaticGeometryCurrent())
            ImGui::TextDisabled("Rebuilding ray tracing and collision...");
        else
            ImGui::TextDisabled("%s", m_Status.c_str());
        ImGui::EndMainMenuBar();
    }

    void EditorLayer::DrawEntitiesPanel()
    {
        ImGui::SetNextWindowPos(ImVec2(12.0f, 36.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280.0f, 330.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Entities"))
        {
            if (m_Entities.empty())
                ImGui::TextDisabled("The scene is empty.\nPick something from Assets to place it.");
            for (size_t i = 0; i < m_Entities.size(); ++i)
            {
                const EntityDescription& description = m_Entities[i].description;
                std::string label = (description.name.empty() ? "(unnamed)" : description.name) + "##" + std::to_string(i);
                if (ImGui::Selectable(label.c_str(), m_Selected == i))
                    m_Selected = i;
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    FocusSelected();
            }
        }
        ImGui::End();
    }

    void EditorLayer::DrawInspector()
    {
        ImGuiIO& io = ImGui::GetIO();
        // Sized to what it shows, growing leftward from the right edge
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 12.0f, 36.0f), ImGuiCond_FirstUseEver, ImVec2(1.0f, 0.0f));
        if (ImGui::Begin("Inspector", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::PushItemWidth(230.0f);
            Entity* selected = Selected();
            if (!selected)
                ImGui::TextDisabled("Nothing selected.\nClick an entity in the scene or in Entities.");
            else
            {
                EntityDescription& description = selected->description;
                // An edit over several frames (typing, dragging) is one undo step
                auto trackEdit = [this] {
                    if (ImGui::IsItemActivated()) BeginEdit();
                    if (ImGui::IsItemDeactivated()) EndEdit();
                };
                bool changed = false;

                std::array<char, 256> name{};
                SetText(name, description.name);
                if (ImGui::InputText("Name", name.data(), name.size()))
                {
                    description.name = name.data();
                    changed = true;
                }
                trackEdit();

                const bool marker = !description.asset;
                if (marker)
                {
                    ImGui::LabelText("Asset", "(marker)");
                    ImGui::SetItemTooltip("A place with a name and no model. The game finds it by its name.");
                }
                else
                {
                    std::filesystem::path asset = description.asset->lexically_relative(std::filesystem::absolute(m_Spec.AssetRoot));
                    ImGui::LabelText("Asset", "%s", (asset.empty() || *asset.begin() == ".." ? *description.asset : asset).generic_string().c_str());
                }

                changed |= ImGui::DragFloat3("Position", &description.position.x, 0.02f, 0.0f, 0.0f, "%.3f");
                trackEdit();
                changed |= ImGui::DragFloat3("Rotation", &description.rotation.x, 0.5f, 0.0f, 0.0f, "%.1f deg");
                trackEdit();
                changed |= ImGui::DragFloat3("Scale", &description.scale.x, 0.01f, 0.001f, 1000.0f, "%.3f");
                trackEdit();

                bool isStatic = description.isStatic;
                ImGui::BeginDisabled(marker); // Nothing to ray trace or collide with
                if (ImGui::Checkbox("Static", &isStatic))
                {
                    RecordUndo();
                    description.isStatic = isStatic;
                    changed = true;
                }
                ImGui::EndDisabled();
                ImGui::SetItemTooltip("Static entities are ray traced and collided with.\nDynamic ones can be moved by the game, but only the raster modes draw them.");

                if (changed)
                    PushToApplication(*selected);

                ImGui::Separator();
                if (ImGui::Button("Duplicate"))
                    DuplicateSelected();
                ImGui::SameLine();
                if (ImGui::Button("Delete"))
                    DeleteSelected();
                ImGui::SameLine();
                if (ImGui::Button("Look at"))
                    FocusSelected();
            }
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }

    void EditorLayer::DrawAssetsPanel()
    {
        ImGui::SetNextWindowPos(ImVec2(12.0f, 378.0f), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(280.0f, 360.0f), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Assets"))
        {
            ImGui::SetNextItemWidth(-70.0f);
            ImGui::InputTextWithHint("##filter", "Filter", m_AssetFilter.data(), m_AssetFilter.size());
            ImGui::SameLine();
            if (ImGui::Button("Rescan"))
                ScanAssets();
            if (m_Assets.empty())
                ImGui::TextDisabled("No glTF files under %s", m_Spec.AssetRoot.generic_string().c_str());

            std::string filter = m_AssetFilter.data();
            std::transform(filter.begin(), filter.end(), filter.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            ImGui::BeginChild("list");
            for (const std::filesystem::path& asset : m_Assets)
            {
                std::string label = asset.generic_string();
                std::string lower = label;
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                if (!filter.empty() && lower.find(filter) == std::string::npos)
                    continue;
                if (ImGui::Selectable(label.c_str(), m_Placing == asset))
                    m_Placing = asset;
                ImGui::SetItemTooltip("Click, then click in the scene to place it");
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }

    void EditorLayer::DrawFileDialogs()
    {
        const char* title = m_FileDialog == FileDialog::Open ? "Open scene" : "Save scene as";
        if (m_FileDialog != FileDialog::None && !ImGui::IsPopupOpen(title))
            ImGui::OpenPopup(title);
        if (m_FileDialog == FileDialog::None)
            return;

        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.4f), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowSize(ImVec2(620.0f, 0.0f), ImGuiCond_Appearing);
        if (!ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_NoSavedSettings))
            return;

        bool open = m_FileDialog == FileDialog::Open;
        ImGui::SetNextItemWidth(-1.0f);
        bool entered = ImGui::InputText("##path", m_FilePath.data(), m_FilePath.size(), ImGuiInputTextFlags_EnterReturnsTrue);
        if (ImGui::IsWindowAppearing())
            ImGui::SetKeyboardFocusHere(-1);
        if (open)
        {
            ImGui::TextDisabled("Scene files under %s:", m_Spec.AssetRoot.generic_string().c_str());
            for (const std::filesystem::path& scene : m_Scenes)
                if (ImGui::Selectable(scene.generic_string().c_str(), false, ImGuiSelectableFlags_NoAutoClosePopups))
                    SetText(m_FilePath, (m_Spec.AssetRoot / scene).generic_string());
        }
        else
            ImGui::TextDisabled("Relative paths are from the working directory. If that's a build\ndirectory, its Assets are overwritten from the source tree on\nevery build: save into the source tree to keep the file.");

        bool confirm = ImGui::Button(open ? "Open" : "Save", ImVec2(120.0f, 0.0f)) || entered;
        ImGui::SameLine();
        bool cancel = ImGui::Button("Cancel", ImVec2(120.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false);
        if (confirm && m_FilePath[0] != '\0')
        {
            std::filesystem::path path = m_FilePath.data();
            open ? OpenScene(path) : SaveAs(path);
        }
        if (confirm || cancel)
        {
            m_FileDialog = FileDialog::None;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}
