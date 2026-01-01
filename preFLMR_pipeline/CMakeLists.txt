# preFLMR Pipeline - Full symlink setup (all changes immediately reflected)

message(STATUS "Setting up preFLMR pipeline with symlinks")

add_custom_target(setup_preflmr_pipeline ALL
    COMMENT "Setting up preFLMR pipeline configuration and files"
)

add_custom_command(TARGET setup_preflmr_pipeline POST_BUILD
    # Create base directory
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline
    
    # 1. Symlink the entire cfg directory first (to get the directory structure)
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/cfg
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg_source
    
    # 2. Copy only the node directories structure (n0-n4) to build
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n0
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n1
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n2
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n3
    COMMAND ${CMAKE_COMMAND} -E make_directory 
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n4
    
    # 3. Symlink template files to cfg root
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/cfg/layout.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/layout.json.tmp
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/cfg/dfgs.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/dfgs.json.tmp
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/cfg/udl_dlls.cfg.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/udl_dlls.cfg.tmp
    COMMAND ln -sf ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/clear_log.sh.tmp
    
    # 4. Symlink python_udls
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
    
    # 5. Symlink client directory
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/client
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/client
    
    # 6. Symlink requirements.txt
    COMMAND ${CMAKE_COMMAND} -E create_symlink 
        ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/requirements.txt
    
    VERBATIM
    COMMENT "Symlinking preFLMR pipeline files"
)

# Create node-specific symlinks (n0-n4)
add_custom_command(TARGET setup_preflmr_pipeline POST_BUILD
    # Node n0
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/layout.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n0/layout.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/dfgs.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n0/dfgs.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/udl_dlls.cfg.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n0/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n0/python_udls
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/clear_log.sh.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n0/clear_log.sh
    
    # Node n1
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/layout.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n1/layout.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/dfgs.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n1/dfgs.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/udl_dlls.cfg.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n1/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n1/python_udls
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/clear_log.sh.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n1/clear_log.sh
    
    # Node n2
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/layout.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n2/layout.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/dfgs.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n2/dfgs.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/udl_dlls.cfg.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n2/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n2/python_udls
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/clear_log.sh.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n2/clear_log.sh
    
    # Node n3
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/layout.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n3/layout.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/dfgs.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n3/dfgs.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/udl_dlls.cfg.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n3/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n3/python_udls
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/clear_log.sh.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n3/clear_log.sh
    
    # Node n4
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/layout.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n4/layout.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/dfgs.json.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n4/dfgs.json
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/udl_dlls.cfg.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n4/udl_dlls.cfg
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n4/python_udls
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/clear_log.sh.tmp
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/n4/clear_log.sh
    
    # Symlink python_udls at cfg root
    COMMAND ln -sf ${CMAKE_BINARY_DIR}/preFLMR_pipeline/python_udls
        ${CMAKE_BINARY_DIR}/preFLMR_pipeline/cfg/python_udls
    
    # Make clear_log.sh executable
    COMMAND chmod 755 ${CMAKE_CURRENT_SOURCE_DIR}/cfg/clear_log.sh.tmp
    
    VERBATIM
    COMMENT "Creating node-specific configuration symlinks"
)

