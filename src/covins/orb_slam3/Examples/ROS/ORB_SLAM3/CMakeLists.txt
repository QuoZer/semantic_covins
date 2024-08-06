cmake_minimum_required(VERSION 2.4.6)
project(ORB_SLAM3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

find_package(catkin REQUIRED COMPONENTS
             covins_comm
             opencv3_catkin
             eigen_catkin
             cv_bridge
             roscpp
             std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

set(EIGEN_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../../../../../devel/include/eigen3)
find_package(Eigen3 3.3.4 EXACT REQUIRED)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../../cmake_modules)

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

#find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../../
${PROJECT_SOURCE_DIR}/../../../include
${PROJECT_SOURCE_DIR}/../../../include/CameraModels
${Pangolin_INCLUDE_DIRS}
${covins_comm_INCLUDE_DIRS}
# include /Thirdparty for cereal
${PROJECT_SOURCE_DIR}/../../../Thirdparty
)

set(LIBS
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../../../../../build_thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../../../../../build_thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../../../../../build_thirdparty/orb_slam3/lib/libORB_SLAM3.so
-lboost_system
${covins_comm_LIBRARIES}
)

# Node for monocular camera
cs_add_executable(Mono
src/ros_mono.cc
)

target_link_libraries(Mono
${LIBS}
)

## Node for monocular camera (Augmented Reality Demo)
#rosbuild_add_executable(MonoAR
#src/AR/ros_mono_ar.cc
#src/AR/ViewerAR.h
#src/AR/ViewerAR.cc
#)

#target_link_libraries(MonoAR
#${LIBS}
#)

## Node for stereo camera
#rosbuild_add_executable(Stereo
#src/ros_stereo.cc
#)

#target_link_libraries(Stereo
#${LIBS}
#)

## Node for RGB-D camera
#rosbuild_add_executable(RGBD
#src/ros_rgbd.cc
#)

#target_link_libraries(RGBD
#${LIBS}
#)

# Node for monocular-inertial camera
cs_add_executable(Mono_Inertial
src/ros_mono_inertial.cc
)

target_link_libraries(Mono_Inertial
${LIBS}
)

# Node for semantic monocular-inertial camera
cs_add_executable(Semantic_Mono_Inertial
src/ros_semantic_mono_inertial.cc
)

target_link_libraries(Semantic_Mono_Inertial
${LIBS}
)

## Node for stereo-inertial camera
#rosbuild_add_executable(Stereo_Inertial
#src/ros_stereo_inertial.cc
#)

#target_link_libraries(Stereo_Inertial
#${LIBS}
#)
