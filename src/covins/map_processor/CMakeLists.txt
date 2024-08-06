cmake_minimum_required(VERSION 2.4.6)
project(map_processor)

find_package(catkin_simple REQUIRED COMPONENTS
            rospy 
            std_msgs)
catkin_simple(ALL_DEPS_REQUIRED)

add_compile_options(-std=c++1z -pthread -pipe -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-unused-function -Wno-vla)

IF(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

catkin_install_python(PROGRAMS scripts/floorplan_extraction_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)