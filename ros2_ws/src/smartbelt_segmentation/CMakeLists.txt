cmake_minimum_required(VERSION 3.8)

project(smartbelt_segmentation)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(pcl_conversions REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(tf2_msgs REQUIRED)

include_directories(include)

# First executable
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

target_link_libraries(mk_matching_ptcld ${PCL_LIBRARIES} /lib/x86_64-linux-gnu/libcurl.so.4)

# Second executable
add_executable(mk_rgb_seg src/MK_rgb_seg.cpp)
ament_target_dependencies(mk_rgb_seg
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

target_link_libraries(mk_rgb_seg ${PCL_LIBRARIES})

# Third executable
add_executable(mk_rgb_seg_bag src/MK_rgb_seg_bag.cpp)
ament_target_dependencies(mk_rgb_seg_bag
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
  rosbag2_cpp
)

target_link_libraries(mk_rgb_seg_bag ${PCL_LIBRARIES} rosbag2_cpp::rosbag2_cpp)

# Fourth executable
add_executable(mk_pc_seg src/MK_pc_seg.cpp)
ament_target_dependencies(mk_pc_seg
  rclcpp
  sensor_msgs
  geometry_msgs
  image_geometry
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  tf2_msgs
  cv_bridge
  OpenCV
  image_transport
  pcl_conversions
  rosbag2_cpp
)

target_link_libraries(mk_pc_seg ${PCL_LIBRARIES} rosbag2_cpp::rosbag2_cpp)

# Install all executables
install(TARGETS
  mk_matching_ptcld
  mk_rgb_seg
  mk_rgb_seg_bag
  mk_pc_seg
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
