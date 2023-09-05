# Copyright 2022-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.8)
project(cvnode_manager)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(BUILD_GUI "Builds GUI for input and output data visualization" OFF)

include_directories(include)

find_package(ament_cmake REQUIRED)
find_package(kenning_computer_vision_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

if(BUILD_TESTING)
    find_package(ament_cmake_clang_format REQUIRED)
    find_package(ament_cmake_clang_tidy REQUIRED)

    ament_clang_format(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
        "include"
        "src"
    )

    ament_clang_tidy(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format
        ${CMAKE_BINARY_DIR}/compile_commands.json
        "include"
        "src"
    )
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    message("nlohmann_json not found - building it")
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

#############

add_library(${PROJECT_NAME} SHARED
    src/cvnode_manager.cpp
)

ament_target_dependencies(${PROJECT_NAME}
    kenning_computer_vision_msgs
    rcl_interfaces
    rclcpp
    rclcpp_components
)

rclcpp_components_register_nodes(${PROJECT_NAME} "cvnode_manager::CVNodeManager")

target_link_libraries(${PROJECT_NAME}
    nlohmann_json::nlohmann_json
)

#############

if (BUILD_GUI)
    add_library(${PROJECT_NAME}_gui SHARED
        src/cvnode_manager_gui.cpp
    )

    find_package(gui_node QUIET)
    if(NOT gui_node_FOUND)
        message("gui_node not found - building it")
        include(FetchContent)
        FetchContent_Declare(gui_node
            GIT_REPOSITORY https://github.com/antmicro/ros2-gui-node
            GIT_TAG main
        )
        FetchContent_MakeAvailable(gui_node)
        target_link_libraries(${PROJECT_NAME}_gui
            gui_node
            gui_node_core_widgets
        )
    else()
        message("gui_node found - using it")
        ament_target_dependencies(${PROJECT_NAME}_gui
            gui_node
        )
    endif()

    find_package(OpenCV REQUIRED)

    target_link_libraries(${PROJECT_NAME}_gui
        ${OpenCV_LIBS}
    )

    ament_target_dependencies(${PROJECT_NAME}_gui
        kenning_computer_vision_msgs
        rclcpp
        rclcpp_components
    )

    rclcpp_components_register_nodes(${PROJECT_NAME}_gui "cvnode_manager::gui::CVNodeManagerGUI")

    install(TARGETS
        ${PROJECT_NAME}_gui
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif(BUILD_GUI)

#############

install(TARGETS
    cvnode_manager
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

ament_package()
