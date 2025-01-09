#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm2

#SBATCH --job-name=full_data_pipe_arm2
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=12:00:00

#SBATCH --ntasks=4         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=13  # max number of cores each task below can use
#SBATCH --mem=640G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task >= total cores used below +3.

#SBATCH --gres=gpu:4       # Request 1 GPU for the job

## list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# echo "working directory = "$SLURM_SUBMIT_DIR

# cd /sailhome/weizhuo2/Documents/Data_pipe/Scripts/

# SCRIPT_NAME=GSAM_labeler.py
SCRIPT_NAME=DINOv2_labeler.py
source /arm/u/weizhuo2/anaconda3/etc/profile.d/conda.sh
conda activate GSAM
srun --gres=gpu:1 -n 1 -c 12 --mem 130G --exclusive  python $SCRIPT_NAME 'V2DataRedo_realsense0801_lag.bag' &
srun --gres=gpu:1 -n 1 -c 12 --mem 165G --exclusive  python $SCRIPT_NAME 'V2DataRedo_realsense0712_lag.bag' &
srun --gres=gpu:1 -n 1 -c 12 --mem 195G --exclusive  python $SCRIPT_NAME 'V2DataRedo_human0306final_lag.bag' &
srun --gres=gpu:1 -n 1 -c 12 --mem 135G --exclusive  python $SCRIPT_NAME 'V2DataRedo_human0401_lag.bag' &
srun --gres=gpu:1 -n 1 -c 12 --mem 115G --exclusive  python $SCRIPT_NAME 'V2DataRedo_human0408_lag.bag' &
srun --gres=gpu:1 -n 1 -c 12 --mem 70G --exclusive  python $SCRIPT_NAME 'V2DataRedo_field_lag.bag' &

# > 7x
wait

echo "Done"