TO RUN:

cd /home/mkhanum/datapipe/ros2_ws
colcon build --packages-select smartbelt_segmentation --cmake-clean-first
source install/setup.bash

Since GSAM2 is slow (need to optimize run_GSAM2.py code), run
ros2 run smartbelt_segmentation mk_rgb_seg_bag <input_bag_path> <output_bag_path>

ex.
ros2 run smartbelt_segmentation mk_rgb_seg_bag /home/mkhanum/datapipe/Bags/stair1 /home/mkhanum/datapipe/Bags/stair1_full

to get segmented bag of Topic: segmented_mask | Type: sensor_msgs/msg/Image

To get segmented pointcloud,
ros2 run smartbelt_segmentation mk_pc_seg 

edit these lines and run:
    reader_mask.open("/home/mkhanum/datapipe/Bags/stair1_full");
    reader_cloud.open("/home/mkhanum/datapipe/Bags/stair1");
    writer_->open("/home/mkhanum/datapipe/Bags/stair1_seg_pc");

you may need to delete output_bag is already there

ex.
ros2 run smartbelt_segmentation mk_pc_seg


to run seg pc, need to run the yaml file
ros2 bag play /home/mkhanum/datapipe/Bags/stair1_seg_pc --qos-profile-overrides-path /home/mkhanum/datapipe/ros2_ws/src/smartbelt_segmentation/scripts/reliability_override.yaml --loop