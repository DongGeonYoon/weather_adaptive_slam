cmake_minimum_required(VERSION 3.8)
project(fast_lio)

option(FAST_LIO_ENABLE_CUDA "Enable CUDA acceleration for rank computation" ON)
set(FAST_LIO_CUDA_AVAILABLE OFF)
if(FAST_LIO_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89" CACHE STRING
            "CUDA architectures for FAST-LIO rank kernels" FORCE)
      endif()
      set(FAST_LIO_CUDA_AVAILABLE ON)
      message(STATUS "FAST-LIO CUDA rank acceleration: ON (${CMAKE_CUDA_COMPILER})")
    else()
      message(WARNING "CUDA compiler found, but CUDAToolkit was not found; using CPU rank")
    endif()
  else()
    message(STATUS "FAST-LIO CUDA rank acceleration: OFF (CUDA compiler not found)")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

ADD_COMPILE_OPTIONS(-std=c++14)
ADD_COMPILE_OPTIONS(-std=c++14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message("Processer number:  ${N}")

  set(FAST_LIO_MP_PROC_NUM "" CACHE STRING "OpenMP thread count for FAST-LIO mapping loops. Empty means auto.")
  if(FAST_LIO_MP_PROC_NUM STREQUAL "")
    if(N GREATER 8)
      math(EXPR FAST_LIO_MP_PROC_NUM "${N} - 2")
    elseif(N GREATER 1)
      math(EXPR FAST_LIO_MP_PROC_NUM "${N} - 1")
    else()
      set(FAST_LIO_MP_PROC_NUM 1)
    endif()
  endif()

  if(FAST_LIO_MP_PROC_NUM GREATER 1)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=${FAST_LIO_MP_PROC_NUM})
    message("core for MP: ${FAST_LIO_MP_PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
    message("core for MP: 1")
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

# ROS dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(dependencies
  rclcpp
  rclcpp_components
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  std_srvs
  pcl_ros
  pcl_conversions
  livox_ros_driver2
)

# Thirdparty libraries
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

message(Eigen: ${EIGEN3_INCLUDE_DIR})
message(STATUS "PCL: ${PCL_INCLUDE_DIRS}")

set(msg_files
  "msg/Pose6D.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
)
ament_export_dependencies(rosidl_default_runtime)

set(FAST_LIO_SOURCES
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
)
if(FAST_LIO_CUDA_AVAILABLE)
  list(APPEND FAST_LIO_SOURCES src/rank_cuda.cu)
endif()

add_executable(fastlio_mapping ${FAST_LIO_SOURCES})
target_include_directories(fastlio_mapping PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(fastlio_mapping ${PCL_LIBRARIES} ${PYTHON_LIBRARIES} Eigen3::Eigen)
target_include_directories(fastlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})
if(FAST_LIO_CUDA_AVAILABLE)
  target_compile_definitions(fastlio_mapping PRIVATE FAST_LIO_USE_CUDA)
  target_link_libraries(fastlio_mapping CUDA::cudart)
  set_target_properties(fastlio_mapping PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

list(APPEND EOL_LIST "foxy" "galactic" "eloquent" "dashing" "crystal")

if($ENV{ROS_DISTRO} IN_LIST EOL_LIST)
  # Custommsg to support foxy & galactic
  rosidl_target_interfaces(fastlio_mapping
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(fastlio_mapping ${cpp_typesupport_target})
endif()

ament_target_dependencies(fastlio_mapping ${dependencies})

# ---------------- Install --------------- #
install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config launch rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS tools/evaluate_weather_adaptive.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY doc
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
