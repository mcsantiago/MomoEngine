# ----------- doctest via FetchContent -----------------
include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
)
FetchContent_MakeAvailable(doctest)

add_executable(MomoEngineTests
  main.cpp
  BvhTests.cpp
  WideBvhTests.cpp
)

target_link_libraries(MomoEngineTests PRIVATE MomoEngine doctest::doctest)

if (MSVC)
  target_compile_options(MomoEngineTests PRIVATE /W4 /permissive-)
else()
  target_compile_options(MomoEngineTests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME MomoEngineTests COMMAND MomoEngineTests)
