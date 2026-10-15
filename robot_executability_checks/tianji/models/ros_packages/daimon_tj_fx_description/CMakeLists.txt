cmake_minimum_required(VERSION 3.10)
project(daimon_tj_fx_description)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(wbc_msgs REQUIRED)

# ---- initial_joint_state_seed_node (C++) ----
add_executable(initial_joint_state_seed_node
  src/initial_joint_state_seed_node.cpp)
target_include_directories(initial_joint_state_seed_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
)
target_link_libraries(initial_joint_state_seed_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
)

# ---- wbc_arm_ik_goal_markers_node ----
add_executable(wbc_arm_ik_goal_markers_node
  src/wbc_arm_ik_goal_markers_node.cpp)
target_include_directories(wbc_arm_ik_goal_markers_node PRIVATE
  ${rclcpp_INCLUDE_DIRS}
  ${sensor_msgs_INCLUDE_DIRS}
  ${visualization_msgs_INCLUDE_DIRS}
  ${geometry_msgs_INCLUDE_DIRS}
  ${wbc_msgs_INCLUDE_DIRS}
)
target_link_libraries(wbc_arm_ik_goal_markers_node
  ${rclcpp_LIBRARIES}
  ${sensor_msgs_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
  ${geometry_msgs_LIBRARIES}
  ${wbc_msgs_LIBRARIES}
  interactive_markers::interactive_markers
  ament_index_cpp::ament_index_cpp
)

# ---- Install ----
install(TARGETS
  initial_joint_state_seed_node
  wbc_arm_ik_goal_markers_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME})

install(FILES README.md
  DESTINATION share/${PROJECT_NAME})

# Python script (ROS 2 version)
install(PROGRAMS
  scripts/initial_joint_state_seed_node.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
