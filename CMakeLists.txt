cmake_minimum_required(VERSION 3.16)
project(longest_ticket CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Static linking: the binary no longer loads libstdc++ etc. at start-up (measured on Linux:
# 2.1 ms -> 1.2 ms per run on the example input). macOS cannot link fully static, so it is skipped there.
option(LONGEST_STATIC "link the runtime statically (Linux / Windows)" ON)
if(LONGEST_STATIC AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

find_package(Threads REQUIRED)
add_executable(longest src/main.cpp)
target_link_libraries(longest PRIVATE Threads::Threads)
if(MSVC)
  target_compile_options(longest PRIVATE /W4 /EHsc /utf-8 $<$<CONFIG:Release>:/O2>)
else()
  target_compile_options(longest PRIVATE -Wall -Wextra $<$<CONFIG:Release>:-O3>)
endif()
if(LONGEST_STATIC AND NOT APPLE AND NOT MSVC)
  target_link_options(longest PRIVATE -static)
endif()
