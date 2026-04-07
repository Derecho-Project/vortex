cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
project(example_udl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT TARGET pyscheduler::pyscheduler)
  find_package(pyscheduler CONFIG REQUIRED)
endif()
find_package(derecho REQUIRED)
find_package(cascade REQUIRED)
find_package(CUDAToolkit QUIET)

add_library(a_udl SHARED a_udl.cpp)
target_include_directories(a_udl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
target_link_libraries(a_udl
    PRIVATE
    derecho::cascade
    pyscheduler::pyscheduler
    CUDA::cudart
    vortex_scheduler
)

add_library(b_udl SHARED b_udl.cpp)
target_include_directories(b_udl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
target_link_libraries(b_udl
    PRIVATE
    derecho::cascade
    pyscheduler::pyscheduler
    CUDA::cudart
    vortex_scheduler
)

add_library(c_udl SHARED c_udl.cpp)
target_include_directories(c_udl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
target_link_libraries(c_udl
    PRIVATE
    derecho::cascade
    pyscheduler::pyscheduler
    CUDA::cudart
    vortex_scheduler
)

add_library(d_udl SHARED d_udl.cpp)
target_include_directories(d_udl PRIVATE
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
)
target_link_libraries(d_udl
    PRIVATE
    derecho::cascade
    pyscheduler::pyscheduler
    CUDA::cudart
    vortex_scheduler
)


add_custom_command(TARGET a_udl POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_CURRENT_BINARY_DIR}/cfg

    COMMAND ln -sfnT ${CMAKE_CURRENT_SOURCE_DIR}/python_udls
        ${CMAKE_BINARY_DIR}/example_scheduler/python_udls

    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/layout.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/layout.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/layout.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/layout.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/dfgs.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/dfgs.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/dfgs.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/jobs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/jobs.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/jobs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/jobs.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/jobs.json.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/jobs.json
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/udl_dlls.cfg
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/udl_dlls.cfg
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/udl_dlls.cfg
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/clear_log.sh
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/clear_log.sh
    COMMAND ln -sfn ${CMAKE_CURRENT_BINARY_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/clear_log.sh

    COMMAND chmod 755 ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp

    COMMAND ln -sfn ${CMAKE_CURRENT_SOURCE_DIR}/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n0/python_udls
    COMMAND ln -sfn ${CMAKE_CURRENT_SOURCE_DIR}/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n1/python_udls
    COMMAND ln -sfn ${CMAKE_CURRENT_SOURCE_DIR}/python_udls
        ${CMAKE_CURRENT_BINARY_DIR}/cfg/n2/python_udls

    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg/jobs.json.tmp
            ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
    COMMENT "prepare example configuration"
)
