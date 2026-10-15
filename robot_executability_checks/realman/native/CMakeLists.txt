cmake_minimum_required(VERSION 3.16)

project(tianji_aik7 LANGUAGES CXX)

# This small shared library deliberately has no ROS, Pinocchio, HPIPM, or
# whole-body-controller dependency.  It packages the Aik7_ow analytic core
# together with a DH model regenerated from this repository's complete
# MuJoCo-imported TJ_FX URDF.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Core PATH_SUFFIXES eigen3)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR
      "Eigen3 headers were not found. Install libeigen3-dev, then rerun ./setup.sh.")
  endif()
endif()

add_library(tianji_aik7 SHARED
  src/aik7_bridge.cpp
  src/current_urdf_model.cpp
  vendor/src/aik7_ow_left_tj.cpp
  vendor/src/aik7_ow_right_tj.cpp
)

target_include_directories(tianji_aik7 PRIVATE vendor/include)
if(Eigen3_FOUND)
  target_link_libraries(tianji_aik7 PRIVATE Eigen3::Eigen)
else()
  target_include_directories(tianji_aik7 PRIVATE "${EIGEN3_INCLUDE_DIR}")
endif()

target_compile_options(tianji_aik7 PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(tianji_aik7 PROPERTIES OUTPUT_NAME "tianji_aik7")
