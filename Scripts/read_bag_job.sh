#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm1

#SBATCH --job-name=add_video_frames
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=6:00:00

#SBATCH --ntasks=6         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=7  # max number of cores each task below can use
#SBATCH --mem=320G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task >= total cores used below +3.

## list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# echo "working directory = "$SLURM_SUBMIT_DIR

# cd /sailhome/weizhuo2/Documents/Data_pipe/Scripts/

SCRIPT_NAME=read_bagV3.py

srun -n 1 -c 6 --mem 50G --exclusive  python $SCRIPT_NAME --fpath='/sailhome/weizhuo2/Documents/Data_pipe/Bags/realsense_2022-08-01-17-25-56_lag.bag' --calib_fac=0.80 &
srun -n 1 -c 6 --mem 50G --exclusive  python $SCRIPT_NAME --fpath='/sailhome/weizhuo2/Documents/Data_pipe/Bags/realsense_2022-07-12-19-26-18_lag.bag' --calib_fac=0.80 &
srun -n 1 -c 6 --mem 50G --exclusive  python $SCRIPT_NAME --fpath='/sailhome/weizhuo2/Documents/Data_pipe/Bags/human_2022-04-08-14-23-31_lag.bag' --calib_fac=0.72 &
srun -n 1 -c 7 --mem 50G --exclusive  python $SCRIPT_NAME --fpath='/sailhome/weizhuo2/Documents/Data_pipe/Bags/human_2022-04-01-16-30-44_lag.bag'  --calib_fac=0.65 &
srun -n 1 -c 7 --mem 60G --exclusive  python $SCRIPT_NAME --fpath='/sailhome/weizhuo2/Documents/Data_pipe/Bags/human_2022-03-06-14-51-06_lag.bag' --calib_fac=0.65 &
srun -n 1 -c 7 --mem 50G --exclusive  python $SCRIPT_NAME --fpath='/sailhome/weizhuo2/Documents/Data_pipe/Bags/field_2021-12-09-16-45-58lag.bag' --calib_fac=0.72 &

wait

echo "Done"