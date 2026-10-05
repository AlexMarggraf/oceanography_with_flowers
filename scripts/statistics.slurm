#!/bin/bash

#SBATCH --job-name=pactcs30_statistics                   #This is the name of your job
#SBATCH --cpus-per-task=1                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G              #This is the memory reserved per core.
#Total memory reserved: 8GB

#SBATCH --time=06:00:00        #This is the time that your task will run
#SBATCH --qos=6hours           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=pactcs30_statistics.o     #These are the STDOUT and STDERR files
#SBATCH --error=pactcs30_statistics.e

#This job runs from the current working directory

#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


#load your required modules below
#################################

source ../.venv/bin/activate

#export your required environment variables below
#################################################

python statistics.py
