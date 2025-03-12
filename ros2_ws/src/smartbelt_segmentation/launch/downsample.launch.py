import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    bag_path = "/home/mkhanum/datapipe/Bags/stair1"
    downsampled_bag_path = "/home/mkhanum/datapipe/Bags/stair1_downsampled"

    topics = [
        "/d455/color/camera_info",
        "/d455/color/image_raw",
        "/d455/depth/camera_info",
        "/d455/depth/color/points"
    ]

    throttle_nodes = [
        Node(
            package="topic_tools",
            executable="throttle",
            arguments=["messages", topic, "20.0"],
            remappings=[(topic, f"{topic}_throttled")],
            output="screen"
        )
        for topic in topics
    ]

    return LaunchDescription([
        # Play the original bag file at normal speed
        ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_path, "--rate", "1.0"],
            output="screen"
        ),

        # Throttle each topic
        *throttle_nodes,

        # Record the downsampled topics
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "-o", downsampled_bag_path] + [f"{topic}_throttled" for topic in topics],
            output="screen"
        ),
    ])
