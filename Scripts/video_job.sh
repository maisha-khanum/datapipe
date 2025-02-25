#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm2

#SBATCH --job-name=add_video_frames
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=2:00:00

#SBATCH --ntasks=5         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=11  # max number of cores each task below can use
#SBATCH --mem=190G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task > total cores used below. Equal will not work, has to be greater

## list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# echo "working directory = "$SLURM_SUBMIT_DIR

# cd /sailhome/weizhuo2/Documents/Data_pipe/Scripts/

source /arm/u/weizhuo2/anaconda3/etc/profile.d/conda.sh
conda activate mmt

# SCRIPT_NAME=add_video_frame.py
SCRIPT_NAME=extract_video.py

srun -n 1 -c 10 --mem 30G --exclusive  python $SCRIPT_NAME 'realsense_2022-08-01-17-25-56_lag.bag' &
srun -n 1 -c 10 --mem 30G --exclusive  python $SCRIPT_NAME 'realsense_2022-07-12-19-26-18_lag.bag' &
srun -n 1 -c 10 --mem 30G --exclusive  python $SCRIPT_NAME 'human_2022-04-08-14-23-31_lag.bag' &
srun -n 1 -c 10 --mem 30G --exclusive  python $SCRIPT_NAME 'human_2022-04-01-16-30-44_lag.bag' &
srun -n 1 -c 10 --mem 40G --exclusive  python $SCRIPT_NAME 'human_2022-03-06-14-51-06_lag.bag' &
# srun -n 1 -c 5 --mem 20G --exclusive  python $SCRIPT_NAME 'field_2021-12-09-16-45-58lag.bag' &

wait

echo "Done"