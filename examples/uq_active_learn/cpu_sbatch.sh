#!/bin/sh -l
  
#SBATCH -A xxx
#SBATCH --partition debug   #wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:30:00
#SBATCH --job-name UQ_cpu
#SBATCH --mail-user=xxx
#SBATCH --mail-type=ALL    # When to send emails (BEGIN, END, FAIL, ALL)

module load anaconda
source activate base
conda activate /anvil/scratch/x-mgoliyad1/conda_env/rose_env

python /anvil/scratch/x-mgoliyad1/ROSE/examples/uq_active_learn/run_me.py