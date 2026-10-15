cmake_minimum_required(VERSION 3.20)
project(dm25pd04_eg_urdf_tianji_20260807)

# ament's helper scripts must use the Python interpreter shipped with ROS.
# This workspace can also have unrelated versioned interpreters in ~/.local/bin.
if(NOT Python3_EXECUTABLE AND EXISTS "/usr/bin/python3")
  set(Python3_EXECUTABLE "/usr/bin/python3")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(ament_cmake REQUIRED)

install(DIRECTORY config meshes rviz urdf
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py")

install(FILES README.md
  DESTINATION share/${PROJECT_NAME})

ament_package()
