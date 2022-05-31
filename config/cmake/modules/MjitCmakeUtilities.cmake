# Copyright (c) 2010-2022, Lawrence Livermore National Security, LLC. Produced
# at the Lawrence Livermore National Laboratory. All Rights reserved. See files
# LICENSE and NOTICE for details. LLNL-CODE-806117.
#
# This file is part of the MFEM library. For more information and source code
# availability visit https://mfem.org.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the BSD-3 license. We welcome feedback and contributions, see file
# CONTRIBUTING.md for details.

# JIT configuration: SOURCES => MJIT => SOURCES
if (MFEM_USE_JIT)

string(ASCII 27 ESC)
message(STATUS "MFEM JIT enabled")

########################
# ADD MJIT EXECUTABLE  #
########################
function(add_mjit_executable)
    add_executable(mjit general/jit/parser.cpp)
endfunction(add_mjit_executable)

#################################
# set_mjit_sources_dependencies #
#################################
function(set_mjit_sources_dependencies TARGET SOURCES)
    #message(NOTICE "${ESC}[1;33m[set_mjit_sources_dependencies] TARGET:${TARGET}${ESC}[m")
    #message(NOTICE "${ESC}[1;33m[set_mjit_sources_dependencies] SOURCES:${SOURCES}:${${SOURCES}}${ESC}[m")
    add_custom_target(${TARGET})
    # 'mjit' all input files from source to binary directory
    foreach(source IN LISTS ${SOURCES})
        #message(NOTICE "${ESC}[1;33m[source] ${source}${ESC}[m")
        get_filename_component(name ${source} NAME)
        #message(NOTICE "${ESC}[1;33m[name] ${name}${ESC}[m")
        get_filename_component(dir ${source} DIRECTORY)
        #message(NOTICE "${ESC}[1;33m[dir] ${dir}${ESC}[m")
        file(RELATIVE_PATH source_path ${CMAKE_CURRENT_SOURCE_DIR} ${dir})
        set(binary_path ${CMAKE_CURRENT_BINARY_DIR}/${relpath})
        #message(NOTICE "\t${ESC}[1;33m${source_path}/${name}${ESC}[m")
        #message(NOTICE "\t${ESC}[1;31m[binary_path] ${binary_path}${ESC}[m")
        #message(NOTICE "\t${ESC}[1;31m[source_path] ${source_path}${ESC}[m")
        #message(NOTICE "\t${ESC}[1;31m[relpath] ${relpath}${ESC}[m")
        set(jit ${CMAKE_CURRENT_BINARY_DIR}/${source_path}/${name})
        add_custom_command(OUTPUT ${jit}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${source_path}
            COMMAND mjit ${source} -o ${jit} DEPENDS mjit ${source})
        set(${TARGET} ${${TARGET}} ${jit})
        # create the dependency name from source_path and name
        string(REPLACE " " "_" source_d ${source_path}/${name})
        string(REPLACE "." "_" source_d ${source_d})
        string(REPLACE "-" "_" source_d ${source_d})
        string(REPLACE "/" "_" source_d ${source_d})
        add_custom_target(${source_d} DEPENDS ${jit})
        #message(NOTICE "\t${ESC}[1;31m${source_d}${ESC}[m")
        add_dependencies(${TARGET} ${source_d})
        set_source_files_properties(${jit} PROPERTIES COMPILE_OPTIONS -I${dir})
    endforeach()
    set(${TARGET} ${${TARGET}} PARENT_SCOPE)
endfunction(set_mjit_sources_dependencies)

################################
# set_mjit_compile_definitions #
################################
function(set_mjit_compile_definitions)
    #message(NOTICE "${ESC}[1;33m[mjit_configure]${ESC}[m")
    #message(NOTICE "\t${ESC}[33m[MFEM_CXX] ${MFEM_CXX}${ESC}[m")
    #message(NOTICE "\t${ESC}[33m[MFEM_EXT_LIBS] ${MFEM_EXT_LIBS}${ESC}[m")
    #message(NOTICE "\t${ESC}[33m[MFEM_BUILD_FLAGS] ${MFEM_BUILD_FLAGS}${ESC}[m")

    if (MFEM_USE_CUDA)
       message("${ESC}[1;32m")
       message(NOTICE "CMAKE_CUDA_HOST_COMPILER: ${CMAKE_CUDA_HOST_COMPILER}")
       message(NOTICE "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
       message(NOTICE "CUDA_FLAGS: ${CUDA_FLAGS}")
       message("${ESC}[m")
       set_source_files_properties(general/jit/parser.cpp PROPERTIES LANGUAGE CUDA)
       #target_compile_options(mjit BEFORE PRIVATE "-Wno-unknown-escape-sequence")
    endif() # MFEM_USE_CUDA

    target_compile_definitions(mjit PRIVATE
           "MFEM_CXX=\"${MFEM_CXX}\""
           "MFEM_EXT_LIBS=\"${MFEM_EXT_LIBS}\""
           "MFEM_BUILD_FLAGS=\"${MFEM_BUILD_FLAGS}\""
           "MFEM_LINK_FLAGS=\"${MFEM_BUILD_FLAGS}\"")

    #target_include_directories(mjit PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_definitions(mjit PRIVATE
      "MFEM_CONFIG_FILE=\"${PROJECT_BINARY_DIR}/config/_config.hpp\"")
endfunction(set_mjit_compile_definitions)

endif() # MFEM_USE_JIT