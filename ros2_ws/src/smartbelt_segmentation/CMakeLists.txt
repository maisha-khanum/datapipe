cmake_minimum_required(VERSION 3.8) # Updated minimum version

project(smartbelt_segmentation)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) # Added REQUIRED
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED) # Added REQUIRED
find_package(PCL REQUIRED COMPONENTS common io)
find_package(pcl_conversions REQUIRED)

include_directories(include)

add_executable(mk_matching_ptcld src/MK_matching_ptcld.cpp)
ament_target_dependencies(mk_matching_ptcld
  rclcpp
  sensor_msgs
  geometry_msgs
  image_geometry
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  cv_bridge
  OpenCV
  image_transport
  pcl_conversions
)

target_link_libraries(mk_matching_ptcld ${PCL_LIBRARIES})
target_link_libraries(mk_matching_ptcld /lib/x86_64-linux-gnu/libcurl.so.4)

install(TARGETS
  mk_matching_ptcld
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()