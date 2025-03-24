#!/bin/bash

#SBATCH --job-name  "rose"
#SBATCH --output    "slurm.rose.out"
#SBATCH --error     "slurm.rose.err"
#SBATCH --partition "cpu"
#SBATCH --time      00:60:00
##SBATCH --ntasks    64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --export    NONE
#SBATCH --account   bblj-delta-cpu

source ~/ve/rct_debug/bin/activate

export RADICAL_PROFILE="TRUE"
export RADICAL_REPORT="TRUE"
export RADICAL_LOG_LVL="DEBUG"

python /u/alsaadi1/RADICAL/ROSE/examples/pool_based_classification/run_me.py
