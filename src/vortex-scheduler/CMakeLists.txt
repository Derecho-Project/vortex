cmake_minimum_required(VERSION 3.12.0)

project(vortex_scheduler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(${PROJECT_NAME} SHARED)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/messages.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/arena.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dag_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/udl_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dye.cpp
)

target_compile_features(${PROJECT_NAME} PUBLIC)
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    derecho::cascade
    pyscheduler::pyscheduler
)


find_package(Catch2 2 REQUIRED)
find_package(benchmark REQUIRED)
include(CTest)
include(Catch)

add_executable(vortex_scheduler_tests 
    tests/test_messages.cpp
    tests/test_arena.cpp
    tests/test_dag_registry.cpp
)
target_link_libraries(vortex_scheduler_tests PRIVATE
    vortex_scheduler
    Catch2::Catch2WithMain
)
catch_discover_tests(vortex_scheduler_tests ADD_TAGS_AS_LABELS)