#!/bin/bash

# Check if at least one parameter is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <bag_name> [--reindex]"
    exit 1
fi

which python
source /arm/u/weizhuo2/anaconda3/etc/profile.d/conda.sh
conda activate data_pipe
which python

# Use the first command-line argument as bag_name
bag_name=$1
reindex_flag=""
    # Check if the second parameter is '--reindex'
if [ "$2" == "--reindex" ]; then
    reindex_flag="--reindex"
fi

# bag_name=V2Data_testpipe.bag
# bag_name=V2DataRedo_realsense0801.bag
fpath=/sailhome/weizhuo2/Documents/Data_pipe/Bags/$bag_name

echo "File path: $fpath"

# prepare the bag file with lag_compensation
    # Extract the file path without the extension
base=${bag_name%.bag}
lag_bag_name="${base}_lag.bag"
lag_fpath=/sailhome/weizhuo2/Documents/Data_pipe/Bags/$lag_bag_name
echo "New bag name: $lag_bag_name"
python lag_compensate.py --fpath $fpath

# read_bag to extract data to training_set format
# Calib fac = 1.0 since save_pc already applied the calibration
python read_bagV3.py --fpath=$lag_fpath --calib_fac=1.00 --window_sz=10

mv "/sailhome/weizhuo2/Documents/Data_pipe/Bags/20HZVZS_${base}_lag" /sailhome/weizhuo2/Documents/Data_pipe/Training_sets/

# before we can run the dinov2 labeler, we need to add back 
# the video frames and depth frames
python add_video_frame.py $lag_bag_name $reindex_flag

# # Finally, run DINOv2 to provide segmentation labels
source /arm/u/weizhuo2/anaconda3/etc/profile.d/conda.sh
conda activate GSAM
which python
python DINOv2_labeler.py $lag_bag_name  # window_sz=9

# Turn dataset into mmap arrays
