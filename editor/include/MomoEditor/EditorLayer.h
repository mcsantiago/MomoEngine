#pragma once

#include <Momo/Core/Application.h>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

// The scene editor, as a layer any application can push.
//
// F1 switches between editing and playing. While editing, the application's
// game layers are paused and the scene's entities can be selected, moved,
// added and removed; playing runs the game on a copy that's thrown away when
// editing resumes, so nothing done while playing sticks.
//
//   Right mouse (held)   fly: mouse looks, WASD moves, Q/E down/up, Shift faster
//   Left click           select the entity under the mouse (or place an asset)
//   W / E / R            move / rotate / scale gizmo; hold Ctrl to snap
//   F                    look at the selection
//   Delete, Ctrl+D       delete, duplicate
//   Ctrl+Z, Ctrl+Y       undo, redo
//   Ctrl+S               save
//   Escape               cancel placing, or deselect
namespace Momo::Editor
{
    struct EditorSpecification
    {
        bool StartEditing = true;
        // Searched for the glTF files and scene files the editor offers
        std::filesystem::path AssetRoot = "Assets";
    };

    class EditorLayer : public Layer
    {
    public:
        explicit EditorLayer(EditorSpecification spec = {}) : m_Spec(std::move(spec)) {}

        void OnAttach() override;
        void OnImGui() override;
        bool UpdatesWhileEditing() const override { return true; }

    private:
        // The scene's saved entities, as position / rotation / scale rather
        // than the matrices the application holds, so the angles shown don't
        // jump between equivalent sets while one is being dragged
        struct Entity
        {
            EntityId id;
            EntityDescription description; // Asset path absolute
        };
        using Snapshot = std::vector<EntityDescription>;

        void SetEditing(bool editing);
        void SyncFromApplication();
        Snapshot TakeSnapshot() const;
        void ApplySnapshot(const Snapshot& snapshot);
        void PushToApplication(const Entity& entity);

        // Undo. An edit that takes many frames (a drag) calls BeginEdit when it
        // starts and EndEdit when it ends, and counts as one step if anything changed.
        void RecordUndo();
        void BeginEdit();
        void EndEdit();
        void Undo();
        void Redo();

        void AddEntity(const std::filesystem::path& asset, glm::vec3 position);
        void DuplicateSelected();
        void DeleteSelected();
        void FocusSelected();

        void NewScene();
        void OpenScene(const std::filesystem::path& path);
        void Save();
        void SaveAs(const std::filesystem::path& path);
        void ScanAssets();

        void HandleShortcuts();
        void UpdateCamera();
        void HandleViewportClick();
        void DrawGizmo();
        void DrawSelectionOutline();
        void DrawMenuBar();
        void DrawEntitiesPanel();
        void DrawInspector();
        void DrawAssetsPanel();
        void DrawFileDialogs();

        // A ray from the camera through the mouse; direction of unit length
        void MouseRay(glm::vec3& origin, glm::vec3& direction) const;
        Entity* Selected() { return m_Selected && *m_Selected < m_Entities.size() ? &m_Entities[*m_Selected] : nullptr; }

        EditorSpecification m_Spec;
        std::vector<Entity> m_Entities;
        std::optional<size_t> m_Selected;
        bool m_Modified = false; // Since the last save or open

        std::vector<Snapshot> m_UndoStack;
        std::vector<Snapshot> m_RedoStack;
        std::optional<Snapshot> m_EditStart;

        enum class GizmoMode { Move, Rotate, Scale };
        GizmoMode m_GizmoMode = GizmoMode::Move;
        bool m_GizmoInUse = false;
        bool m_Flying = false;

        // While playing: what editing resumes with
        std::optional<Snapshot> m_PlaySnapshot;
        glm::vec3 m_EditCameraPosition{ 0.0f };
        glm::vec3 m_EditCameraFront{ 0.0f, 0.0f, -1.0f };

        std::vector<std::filesystem::path> m_Assets; // glTF files, relative to the asset root
        std::vector<std::filesystem::path> m_Scenes; // Scene files, likewise
        std::array<char, 128> m_AssetFilter{};
        std::optional<std::filesystem::path> m_Placing; // Asset the next click in the scene places

        enum class FileDialog { None, Open, SaveAs };
        FileDialog m_FileDialog = FileDialog::None;
        std::array<char, 1024> m_FilePath{};
        std::string m_Status; // Last thing that happened, shown in the menu bar
    };
}
