import launch
import launch_ros.actions
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    bag_file_arg = DeclareLaunchArgument(
        "bag_file", default_value="/home/mkhanum/datapipe/Bags/stair1",
        description="Path to the ROS 2 bag file"
    )

    rosbag_player = ExecuteProcess(
        cmd=["ros2", "bag", "play", LaunchConfiguration("bag_file"),
            "--topics", "/d455/color/image_raw"],  # Specify the relevant topics
            # "--read-ahead-queue-size", "1000"],  # Play for 60 seconds, adjust as needed
        output="screen"
    )
    
    ros2_topic_echo = ExecuteProcess(
        cmd=["ros2", "topic", "echo", "/d455/color/image_raw", "-n", "1"],  # Publish a count of messages
        output="screen"
    )

    mk_rgb_seg = launch_ros.actions.Node(
        package="smartbelt_segmentation", executable="mk_rgb_seg",
        name="mk_rgb_seg_node",
        output="screen",
        parameters=[
            {"pt_srv_color_img_topic": "/d455/color/image_raw"}
        ]
    )

    rosbag_recorder = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", "/home/mkhanum/datapipe/Bags/processed_stair1", "/processed_d455/image_gsam2_mask"],
        # cmd=["ros2", "bag", "record", "--overwrite", "-o", "/home/mkhanum/datapipe/Bags/processed_stair1", "/processed_d455/image_gsam2_mask"],       
        output="screen"
    )

    return LaunchDescription([
        bag_file_arg,
        rosbag_player,
        mk_rgb_seg,
        rosbag_recorder
    ])
