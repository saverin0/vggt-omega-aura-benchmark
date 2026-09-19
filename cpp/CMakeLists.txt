# Build description for the C++ geometry core.
#
# CMake does not compile anything itself. It reads this file, finds the compiler
# and the libraries on THIS machine, and writes out the real build commands
# (Makefiles on Linux). That is why the build is two steps:
#
#   cmake -S cpp -B cpp/build      # configure: inspect the machine, generate build files
#   cmake --build cpp/build        # build: run the compiler
#
# python -m vggt_aura.cpp_build runs both, and tells CMake where pybind11 lives.
cmake_minimum_required(VERSION 3.18)
project(vggt_geom_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)        # optimised. "Debug" would be several times slower.
endif()

# The Python to build for, and pybind11 (header-only; installed with pip).
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# One Python extension module from one source file. The geometry itself is
# header-only and lives in include/, with no dependencies at all.
pybind11_add_module(vggt_geom_cpp src/bindings.cpp)
target_include_directories(vggt_geom_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -ffp-contract=off: never fuse a*b+c into one rounded step. NumPy does not, and the
  # differential tests compare against NumPy number for number.
  target_compile_options(vggt_geom_cpp PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

# Put the finished module straight into the build folder, where cpp_build.py looks for it.
set_target_properties(vggt_geom_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
