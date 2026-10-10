cmake_minimum_required(VERSION 3.26) # Updated to include copy_directory_if_different command
project(KHClone LANGUAGES CXX)

# Modern C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json for clangd / nvim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MOMO_BUILD_TESTS "Build the engine unit tests" ON)
if (MOMO_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(engine)
add_subdirectory(application)