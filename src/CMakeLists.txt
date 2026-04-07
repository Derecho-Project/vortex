# Build Python UDL Manager library
find_package(Python3 3.10.0 QUIET COMPONENTS Interpreter Development Development.Module Development.Embed NumPy)
    
add_subdirectory(scheduler-components)

if (Python3_FOUND)
    message(STATUS "Python3 found: Building python_udl_manager")
    
    # Configure header file
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in 
        ${CMAKE_CURRENT_BINARY_DIR}/config.h
    )

    add_subdirectory(python-udl-interface)

    # Build shared library
    add_library(python_udl_manager SHARED 
        ${CMAKE_CURRENT_SOURCE_DIR}/python_udl_manager.cpp
    )
    
    target_include_directories(python_udl_manager PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    )

    target_link_libraries(python_udl_manager 
        cascade 
        Python3::Python 
        Python3::NumPy
    )

else()
    message(WARNING "Python3 not found. python_udl_manager will not be built.")
endif()