#!/bin/bash

## Job Settings
#SBATCH --account=arm
#SBATCH --partition=arm
#SBATCH --nodelist=arm1

#SBATCH --job-name=full_data_pipe
#SBATCH --output=../Jobs/job_%j_%N.out
# SBATCH --error=../Jobs/job_%j_%N.err
#SBATCH --time=48:00:00

#SBATCH --ntasks=2         # total number of tasks that will run in parallel
#SBATCH --cpus-per-task=14  # max number of cores each task below can use
#SBATCH --mem=330G          # total number of mem this while batch can have
# You need to make sure ntasks*cpus_per_task >= total cores used below +3.

#SBATCH --gres=gpu:2       # Request 1 GPU for the job

## list out some useful information
# echo "SLURM_JOBID="$SLURM_JOBID
# echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
# echo "SLURM_NNODES"=$SLURM_NNODES
# echo "SLURMTMPDIR="$SLURMTMPDIR
# echo "working directory = "$SLURM_SUBMIT_DIR

# cd /sailhome/weizhuo2/Documents/Data_pipe/Scripts/

# srun -n 1 -c 6 --mem 50G --exclusive ./full_datapipe.sh V2DataRedo_realsense0801.bag &
# srun -n 1 -c 6 --mem 50G --exclusive ./full_datapipe.sh V2DataRedo_realsense0712.bag &
# srun -n 1 -c 6 --mem 50G --exclusive ./full_datapipe.sh V2DataRedo_human0408.bag &
# srun -n 1 -c 7 --mem 50G --exclusive ./full_datapipe.sh V2DataRedo_human0401.bag &
# srun -n 1 -c 7 --mem 60G --exclusive ./full_datapipe.sh V2DataRedo_human0306final.bag &
# srun -n 1 -c 7 --mem 50G --exclusive ./full_datapipe.sh V2DataRedo_field.bag &

# srun --gres=gpu:1 -n 1 -c 12 --mem 180G --exclusive ./full_datapipe.sh V2DataNew_231226psyc1.bag &
# srun --gres=gpu:1 -n 1 -c 12 --mem 180G --exclusive ./full_datapipe.sh V2DataNew_231226psyc2.bag &

# srun --gres=gpu:1 -n 1 -c 12 --mem 160G ./full_datapipe.sh V2DataNew_231228lawnew.bag &
# srun --gres=gpu:1 -n 1 -c 12 --mem 160G ./full_datapipe.sh V2DataNew_231228bachtel.bag &
# srun --gres=gpu:1 -n 1 -c 12 --mem 160G ./full_datapipe.sh V2DataNew_231228lawstair.bag &
# srun --gres=gpu:1 -n 1 -c 12 --mem 160G ./full_datapipe.sh V2DataNew_231228quad.bag &
# srun --gres=gpu:1 -n 1 -c 12 --mem 160G ./full_datapipe.sh V2DataNew_231228lawold_needpanofix.bag &

srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231226psyc1C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231226psyc2C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231228bachtelC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231228lawnewC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231228lawold_needpanofixC.bag  --reindex &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231228lawstairC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231228quadC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231230frat1C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231230frat2C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_231231ccrma1C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240103dorm_rainC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240103flomoC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240104darkC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240104dormC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240104hospital1C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240104hospital2C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240104lomita_needpanofixC.bag --reindex &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240105cactusC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240105clarkC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240105clark_shortC.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240105hospitalday1C.bag &
srun --gres=gpu:1 -n 1 -c 12 --mem 160G --exclusive ./full_datapipe.sh V2DataNew_240105hospitalday2C.bag &

wait

echo "Done"