cmake_minimum_required(VERSION 3.5)
project(ros2_ws)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(message_filters REQUIRED)

# Add executable
add_executable(mk_matching_ptcld src/MK_matching_ptcld.cpp)

ament_target_dependencies(mk_matching_ptcld rclcpp sensor_msgs std_msgs message_filters)

install(TARGETS
  mk_matching_ptcld
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
