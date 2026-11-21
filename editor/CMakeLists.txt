# The scene editor: a library holding the editor's layer, which any
# application can push to get edit mode, and a standalone editor built on it.

include(FetchContent)
# ----------- ImGuizmo (move/rotate/scale gizmo) via FetchContent -----------------
# Only downloaded: its own CMakeLists builds extra widgets and expects to
# find Dear ImGui itself, so ImGuizmo.cpp is compiled into the library below.
FetchContent_Declare(
  imguizmo
  GIT_REPOSITORY https://github.com/CedricGuillemet/ImGuizmo.git
  GIT_TAG 105e4fde45e4e74aae8973d98750820ef7aca7e2
)
FetchContent_GetProperties(imguizmo)
if (NOT imguizmo_POPULATED)
  FetchContent_Populate(imguizmo)
endif()

add_library(MomoEditor STATIC
  include/MomoEditor/EditorLayer.h
  src/EditorLayer.cpp
  ${imguizmo_SOURCE_DIR}/src/ImGuizmo.cpp
)
target_include_directories(MomoEditor
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${imguizmo_SOURCE_DIR}/src
)
target_link_libraries(MomoEditor PUBLIC MomoEngine PRIVATE imgui)
if (MSVC)
  set_source_files_properties(src/EditorLayer.cpp PROPERTIES COMPILE_OPTIONS "/W4;/permissive-")
else()
  set_source_files_properties(src/EditorLayer.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Wpedantic")
endif()

# The standalone editor. Run it from the binary dir, like the application,
# so the relative "shaders/" and "Assets/" paths resolve:
#   cd build/clang-release && ./editor/MomoEditor --scene=Assets/Scenes/My.scene.json
add_executable(MomoEditorApp app/main.cpp)
set_target_properties(MomoEditorApp PROPERTIES OUTPUT_NAME "MomoEditor")
target_link_libraries(MomoEditorApp PRIVATE MomoEditor)
