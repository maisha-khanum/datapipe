cmake_minimum_required(VERSION 3.5)
project(smartbelt_segmentation)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport)

include_directories(include)

add_executable(mk_matching_ptcld src/MK_matching_ptcld.cpp)
ament_target_dependencies(mk_matching_ptcld rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_eigen tf2_geometry_msgs cv_bridge OpenCV image_transport)

install(TARGETS
  mk_matching_ptcld
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
