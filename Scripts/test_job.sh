#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm1

#SBATCH --job-name=add_video_frames
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=2:00:00

#SBATCH --ntasks=6         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=6  # max number of cores each task below can use
#SBATCH --mem=320G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task > total cores used below. Equal will not work, has to be greater

## list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# echo "working directory = "$SLURM_SUBMIT_DIR

# cd /sailhome/weizhuo2/Documents/Data_pipe/Scripts/
source /arm/u/weizhuo2/anaconda3/etc/profile.d/conda.sh
conda activate GSAM
srun -n 1 -c 5 --mem 50G --exclusive  which python &
conda activate mmt
srun -n 1 -c 5 --mem 50G --exclusive  which python &
# srun -n 1 -c 5 --mem 50G --exclusive  python add_video_frame.py 'human_2022-04-08-14-23-31_lag.bag' &
# srun -n 1 -c 5 --mem 50G --exclusive  python add_video_frame.py 'human_2022-04-01-16-30-44_lag.bag' &
# srun -n 1 -c 5 --mem 60G --exclusive  python add_video_frame.py 'human_2022-03-06-14-51-06_lag.bag' &
# srun -n 1 -c 5 --mem 50G --exclusive  python add_video_frame.py 'field_2021-12-09-16-45-58lag.bag' &

# srun -n 1 -c 2 --mem 16G --exclusive sleep 10 &     # create job with 1 task
# srun -n 1 -c 2 --mem 1G --exclusive sleep 10 &
# srun -n 1 -c 2 --mem 1G --exclusive sleep 10 &
# srun -n 1 -c 2 --mem 1G --exclusive sleep 10 &
# srun -n 1 -c 2 --mem 1G --exclusive sleep 10 &

# srun -n 1 -c 1 --exclusive sleep 10 &
# srun -n 1 -c 1 --exclusive sleep 10 &
# srun -n 1 -c 1 --exclusive sleep 10 &
# srun -n 1 -c 1 --exclusive sleep 10 &
# srun -n 1 -c 1 --mem 1G --exclusive sleep 10 &
wait

echo "Done"

sacct -j6563673 --format=JobID,Start,End,Elapsed,NCPUS,ReqMem,ExitCode