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
        cmd=["ros2", "bag", "play", LaunchConfiguration("bag_file")],
        output="screen"
    )

    mk_matching_ptcld = launch_ros.actions.Node(
        package="smartbelt_segmentation", executable="mk_matching_ptcld",
        name="mk_matching_ptcld_node",
        output="screen",
        parameters=[
            {"pt_srv_color_img_topic": "/d455/color/image_raw"},
            {"pt_srv_depth_img_cam_info_topic": "/d455/depth/camera_info"},
            {"pt_srv_reg_pt_cld_topic": "/d455/depth/color/points"}
        ]
    )

    rosbag_recorder = ExecuteProcess(
        cmd=["ros2", "bag", "record", "-o", "/home/mkhanum/datapipe/Bags/processed_stair1", "/processed_d455/image_gsam2_mask", "/processed_d455/points_gsam2_mask"],
        output="screen"
    )

    return LaunchDescription([
        bag_file_arg,
        rosbag_player,
        mk_matching_ptcld,
        rosbag_recorder
    ])
