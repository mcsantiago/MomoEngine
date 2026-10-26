cmake_minimum_required(VERSION 3.26) # Updated to include copy_directory_if_different command
project(KHClone LANGUAGES CXX)

# Modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd / nvim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MOMO_BUILD_TESTS "Build the engine unit tests" ON)
# Embeds shader source line info (slangc -g) so GPU profilers such as Nsight
# Graphics can attribute samples to source lines. Off for normal builds.
option(MOMO_SHADER_DEBUG_INFO "Compile shaders with debug info for GPU profiling" OFF)
if (MOMO_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(engine)
add_subdirectory(application)