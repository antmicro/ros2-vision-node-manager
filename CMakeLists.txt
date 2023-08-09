cmake_minimum_required(VERSION 3.8)
project(cvnode_manager)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include_directories(include)

find_package(ament_cmake REQUIRED)
find_package(kenning_computer_vision_msgs REQUIRED)
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

add_library(cvnode_manager SHARED
    src/cvnode_manager.cpp
)

ament_target_dependencies(cvnode_manager
    "kenning_computer_vision_msgs"
    "rclcpp"
    "rclcpp_components"
)

rclcpp_components_register_nodes(cvnode_manager "cvnode_manager::CVNodeManager")

target_link_libraries(cvnode_manager
    nlohmann_json::nlohmann_json
)

#############

install(TARGETS
    cvnode_manager
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

ament_package()
