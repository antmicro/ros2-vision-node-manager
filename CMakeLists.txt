cmake_minimum_required(VERSION 3.8)
project(cvnode_manager)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(cvnode_msgs REQUIRED)
find_package(rclcpp REQUIRED)

include_directories(include)

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

#############

add_library(cvnode_manager SHARED
    src/cvnode_manager.cpp
)

ament_target_dependencies(cvnode_manager
    "cvnode_msgs"
    "rclcpp"
)

#############

install(TARGETS
    cvnode_manager
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

ament_package()
