#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
# SBATCH --nodelist=arm2

#SBATCH --job-name=full_data_job
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=12:00:00

#SBATCH --ntasks=1         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=20  # max number of cores each task below can use
#SBATCH --mem=200G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task >= total cores used below +3.

#SBATCH --gres=gpu:1       # Request 1 GPU for the job

echo $1
./full_datapipe.sh $1
echo "Done"