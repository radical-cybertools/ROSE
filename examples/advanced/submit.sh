#!/bin/bash

#SBATCH --job-name  "rose"
#SBATCH --output    "slurm.rose.out"
#SBATCH --error     "slurm.rose.err"
#SBATCH --partition "cpu"
#SBATCH --time      00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --export    NONE
#SBATCH --account   bblj-delta-cpu

source ~/ve/rct_debug/bin/activate

export RADICAL_PROFILE="TRUE"
export RADICAL_REPORT="TRUE"
export RADICAL_LOG_LVL="DEBUG"


python /u/alsaadi1/RADICAL/ROSE/examples/advanced/run_me.py
