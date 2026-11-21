# Builds the main app for now. Later you can link an engine lib here.
add_executable(MomoApp
    src/main.cpp
)

if (WIN32)
  target_compile_definitions(MomoApp PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Warnings
if (MSVC)
  target_compile_options(MomoApp PRIVATE /W4 /permissive-)
else()
  target_compile_options(MomoApp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Nice output name
set_target_properties(MomoApp PROPERTIES OUTPUT_NAME "HyperbolicTimeChamber")

target_link_libraries(MomoApp PRIVATE MomoEngine)

# Copy game assets next to the compiled shaders so relative
# "Assets/..." paths resolve from the binary dir. Re-runs every build,
# but only copies files that changed.
add_custom_target(MomoAssets ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/Assets
          ${CMAKE_BINARY_DIR}/Assets
  COMMENT "Copying game assets"
)
add_dependencies(MomoApp MomoAssets)

# With the editor built, the application gets edit mode (F1, or --editor to
# start in it), and the standalone editor finds the same assets
if (TARGET MomoEditor)
  target_link_libraries(MomoApp PRIVATE MomoEditor)
  target_compile_definitions(MomoApp PRIVATE MOMO_EDITOR)
  add_dependencies(MomoEditorApp MomoAssets)
endif()
# Convenience target: build, then run from the binary dir so the relative
# "shaders/*.spv" paths in VulkanRenderer resolve.
#   cmake --build build/clang-debug --target run
add_custom_target(run
  COMMAND $<TARGET_FILE:MomoApp>
  DEPENDS MomoApp
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  USES_TERMINAL
  COMMENT "Running The Hyperbolic Time Chamber"
)
