# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "inference_engine_legacy")

set(PUBLIC_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp)
file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADERS_DIR}/*.hpp
                                 ${PUBLIC_HEADERS_DIR}/*.h)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj

source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${PUBLIC_HEADERS})

if(ENABLE_V7_SERIALIZE)
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/cnn_network_impl.cpp"
        PROPERTIES COMPILE_DEFINITIONS ENABLE_V7_SERIALIZE)
endif()

# Create object library

add_library(${TARGET_NAME}_obj OBJECT EXCLUDE_FROM_ALL
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

ie_faster_build(${TARGET_NAME}_obj
    PCH PRIVATE "src/precomp.hpp")

target_include_directories(${TARGET_NAME}_obj PRIVATE
    ${PUBLIC_HEADERS_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src # For CNNNetworkNGraphImpl
    $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:openvino::pugixml,INTERFACE_INCLUDE_DIRECTORIES>)

target_compile_definitions(${TARGET_NAME}_obj PRIVATE $<TARGET_PROPERTY:ngraph,INTERFACE_COMPILE_DEFINITIONS>)

target_link_libraries(${TARGET_NAME}_obj PRIVATE openvino::itt)

add_clang_format_target(${TARGET_NAME}_obj_clang FOR_TARGETS ${TARGET_NAME}_obj)

# Create static library

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL
    $<TARGET_OBJECTS:${TARGET_NAME}_obj>)

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime
                                     PRIVATE openvino::pugixml openvino::itt openvino::runtime::dev)

target_include_directories(${TARGET_NAME} INTERFACE
    $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>)

# Compile with USE_STATIC_IE

set(LEGACY_LIBRARY_SHARED_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/convert_opset1_to_legacy/convert_one_hot_to_one_hot_ie.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/transformations/convert_opset1_to_legacy/convert_nms_5_to_legacy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ngraph_ops/nms_ie.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ngraph_ops/onehot_ie.cpp")
list(REMOVE_ITEM LIBRARY_SRC ${LEGACY_LIBRARY_SHARED_SRCS})

add_library(${TARGET_NAME}_s STATIC EXCLUDE_FROM_ALL
            ${LIBRARY_SRC}
            ${PUBLIC_HEADERS})

target_compile_definitions(${TARGET_NAME}_s PUBLIC USE_STATIC_IE)

target_include_directories(${TARGET_NAME}_s
    PUBLIC
        $<BUILD_INTERFACE:${PUBLIC_HEADERS_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        $<TARGET_PROPERTY:inference_engine_obj,SOURCE_DIR>/src # For CNNNetworkNGraphImpl
        $<TARGET_PROPERTY:openvino::runtime::dev,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino::pugixml,INTERFACE_INCLUDE_DIRECTORIES>)

set_target_properties(${TARGET_NAME}_s PROPERTIES
    EXCLUDE_FROM_ALL ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# LTO

set_target_properties(${TARGET_NAME} ${TARGET_NAME}_obj
                      PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# developer package

openvino_developer_export_targets(COMPONENT core_legacy TARGETS ${TARGET_NAME})

# install

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE} OPTIONAL)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
