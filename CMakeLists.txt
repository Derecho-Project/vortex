cmake_minimum_required(VERSION 3.12.0)

project(vortex LANGUAGES CXX)

# ---- Global defaults ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Default build type for single-config generators
if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Common dependencies ----
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(CURL REQUIRED)
find_package(spdlog 1.3.1 REQUIRED)
find_package(derecho CONFIG REQUIRED)
find_package(cascade CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)
if (NOT DEFINED CMAKE_INSTALL_BINDIR)
    set(CMAKE_INSTALL_BINDIR bin)
endif()

if (NOT DEFINED CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
endif()

# ---- Pipeline selection options ----
option(VORTEX_BUILD_VECTOR_SEARCH_PIPELINE "Build vector_search_pipeline" OFF)

# ---- Always build core components ----
# Python UDL manager (shared library)
add_subdirectory(src)

# preFLMR pipeline (Python files + configuration)
add_subdirectory(preFLMR_pipeline)

# ---- Optional pipelines ----
if (VORTEX_BUILD_VECTOR_SEARCH_PIPELINE)
  message(STATUS "Building vector_search_pipeline")
  add_subdirectory(vector_search_pipeline)
endif()