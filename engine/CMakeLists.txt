# Build the engine as a static lib
add_library(MomoEngine STATIC
  src/Core/Application.cpp
  src/Logging/Logger.cpp
  src/Platform/GLFW/GLFWWindow.cpp
  src/Platform/GLFW/WindowVulkan.cpp
  src/Renderer/VulkanRenderer.cpp
  include/Momo/Renderer/DescriptorAllocator.h
  src/Renderer/DescriptorAllocator.cpp
  include/Momo/Renderer/Camera.h
  src/Renderer/Camera.cpp
  include/Momo/Input/InputState.h
  include/Momo/Input/Key.h
  include/Momo/Assets/ModelLoader.h
  src/Assets/ModelLoader.cpp
  include/Momo/Renderer/VulkanMeshData.h
  include/Momo/Geometry/Vertex.h
  include/Momo/Assets/AssetPool.h
  include/Momo/Assets/Handle.h
  include/Momo/Assets/AssetRegistry.h
  src/Assets/AssetRegistry.cpp
  src/Assets/StbImage.cpp
  include/Momo/Assets/ModelData.h
  include/Momo/Assets/ModelSource.h
  include/Momo/Cache/DataCache.h
  include/Momo/Cache/DataCache.h
  include/Momo/Core/Scene.h
)

# Public headers are under engine/include
target_include_directories(MomoEngine
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(FetchContent)
# ----------- FastGltf via FetchContent -----------------
FetchContent_Declare(
  fastgltf
  GIT_REPOSITORY https://github.com/spnda/fastgltf
  GIT_TAG v0.9.0
)

# ----------- GLFW via FetchContent -----------------
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4
)

# ----------- stb (image decoding) via FetchContent -----------------
# Header-only and has no CMakeLists, so this only downloads the headers; the
# implementation is compiled once in src/Assets/StbImage.cpp.
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
)
FetchContent_MakeAvailable(fastgltf glfw stb)

target_include_directories(MomoEngine PRIVATE ${stb_SOURCE_DIR})

# target_compile_definitions(MomoEngine PRIVATE glfw)

# ----------- Vulkan SDK / Loader -----------------
# Help FindVulkan on Windows if VULKAN_SDK is set
if (WIN32 AND DEFINED ENV{VULKAN_SDK})
  # Typical SDK layout: C:\VulkanSDK\1.x.x.x\{Include,Lib}
  set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/Include" CACHE PATH "")
  # Use the 64-bit lib; if you're targeting Win32, switch to /Lib32/
  set(Vulkan_LIBRARY     "$ENV{VULKAN_SDK}/Lib/vulkan-1.lib" CACHE FILEPATH "")
endif()

# Requires Vulkan SDK installed (LunarG). On macOS this is MoltenVK via the SDK.
# CMake's FindVulkan provides Vulkan::Vulkan if found.
find_package(Vulkan REQUIRED)

# Link the Vulkan loader and GLFW
# Use Vulkan::Vulkan to be cross-platform (Win: vulkan-1.lib, Linux: libvulkan.so, macOS: MoltenVK dylib)
# Vulkan is PUBLIC because VulkanRenderer.h is included in public headers (Application.h)
target_link_libraries(MomoEngine
  PUBLIC
    Vulkan::Vulkan
  PRIVATE
    glfw
    fastgltf::fastgltf
)

# If your GLFW code includes <GLFW/glfw3.h> and you want it to include <vulkan/vulkan.h> for you:
# You can define this globally for the engine sources:
target_compile_definitions(MomoEngine PRIVATE GLFW_INCLUDE_VULKAN)

message(STATUS "Vulkan include: ${Vulkan_INCLUDE_DIRS}")
message(STATUS "Vulkan libs: ${Vulkan_LIBRARIES}")

# Windows-friendly defines (avoid <windows.h> min/max macro conflicts if you include it later)
if (WIN32)
  target_compile_definitions(MomoEngine PRIVATE NOMINMAX)
endif()

# Warnings
if (MSVC)
  target_compile_options(MomoEngine PRIVATE /W4 /permissive-)
else()
  target_compile_options(MomoEngine PRIVATE -Wall -Wextra -Wpedantic)
endif()

# (Add dependencies here later, e.g., glfw/volk)
# target_link_libraries(MomoEngine PUBLIC glfw volk)

# ----------- Shader Compilation with Slang -----------------
# Find slangc compiler (from Vulkan SDK or system PATH)
find_program(SLANGC_EXECUTABLE
  NAMES slangc slangc.exe
  PATHS
    "$ENV{VULKAN_SDK}/Bin"
    "$ENV{SLANG_DIR}/bin"
  DOC "Path to slangc compiler"
)

if(NOT SLANGC_EXECUTABLE)
  message(FATAL_ERROR "slangc not found. Please install Slang or set SLANG_DIR environment variable.")
endif()

message(STATUS "Found slangc: ${SLANGC_EXECUTABLE}")

# Shader source and output directories
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders)

# Ensure output directory exists
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

# Define shader compilation function
function(compile_slang_shader SHADER_SOURCE)
  get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)

  # Output files for vertex and fragment stages
  set(VERT_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.vert.spv)
  set(FRAG_OUTPUT ${SHADER_OUTPUT_DIR}/${SHADER_NAME}.frag.spv)

  # Compile vertex shader
  add_custom_command(
    OUTPUT ${VERT_OUTPUT}
    COMMAND ${SLANGC_EXECUTABLE}
      -target spirv
      -profile spirv_1_5
      -emit-spirv-directly
      -matrix-layout-column-major
      -stage vertex
      -entry vertMain
      -fvk-use-entrypoint-name
      -o ${VERT_OUTPUT}
      ${SHADER_SOURCE}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling vertex shader: ${SHADER_NAME}.vert.spv"
    VERBATIM
  )

  # Compile fragment shader
  add_custom_command(
    OUTPUT ${FRAG_OUTPUT}
    COMMAND ${SLANGC_EXECUTABLE}
      -target spirv
      -profile spirv_1_5
      -emit-spirv-directly
      -matrix-layout-column-major
      -stage fragment
      -entry fragMain
      -fvk-use-entrypoint-name
      -o ${FRAG_OUTPUT}
      ${SHADER_SOURCE}
    DEPENDS ${SHADER_SOURCE}
    COMMENT "Compiling fragment shader: ${SHADER_NAME}.frag.spv"
    VERBATIM
  )

  # Add to parent scope
  set(COMPILED_SHADERS ${COMPILED_SHADERS} ${VERT_OUTPUT} ${FRAG_OUTPUT} PARENT_SCOPE)
endfunction()

# Compile triangle.slang
set(COMPILED_SHADERS "")
compile_slang_shader(${SHADER_SOURCE_DIR}/common/triangle.slang)

# Create custom target for all shaders
add_custom_target(MomoShaders ALL
  DEPENDS ${COMPILED_SHADERS}
  COMMENT "Building engine shaders"
)

# Make the engine depend on shaders being built
add_dependencies(MomoEngine MomoShaders)

# Install shaders to output directory (optional, for deployment)
install(DIRECTORY ${SHADER_OUTPUT_DIR}/
  DESTINATION shaders
  FILES_MATCHING PATTERN "*.spv"
)
