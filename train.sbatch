#!/bin/bash

#SBATCH --job-name=fno_test                   #This is the name of your job
#SBATCH --cpus-per-task=4                 #This is the number of cores reserved
#SBATCH --mem-per-cpu=64G              #This is the memory reserved per core.

#SBATCH --time=00:30:00        #This is the time that your task will run
#SBATCH --qos=a100-30min      #You will run in this queue
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=train_out     #These are the STDOUT and STDERR files
#SBATCH --error=train_err

#This job runs from the current working directory

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


#load your required modules below
#################################


#export your required environment variables below
#################################################


#add your command lines below
#############################
uv run oceanography_with_flowers/train.py --model-path config/fno.yaml --data-path config/pactcs30.yaml
