#!/bin/bash

RP_PROJECT=$1
if test -z "$RP_PROJECT"; then
  RP_PROJECT="PHY122"
fi

N_NODES=$2
if test -z "$N_NODES"; then
  N_NODES=4
fi

RUN_DIR=$PWD
RCT_CONFIG_DIR="${RUN_DIR}/.radical/pilot/configs"
mkdir -p "${RCT_CONFIG_DIR}"
# by default we use 56 cores per node, 8 cores reserved for system process
# TO BE UPDATED IF NEEDED
cat > "${RCT_CONFIG_DIR}/resource_ornl.json" <<EOT
{
    "frontier" : {
        "system_architecture" : {
            "smt"           : 1,
            "blocked_cores" : [0, 8, 16, 24, 32, 40, 48, 56],
            "options"       : ["nvme"]}
    }
}
EOT

sbatch <<EOT
#!/bin/sh

#SBATCH --account          $RP_PROJECT
#SBATCH --partition        batch
#SBATCH --nodes            $N_NODES
#SBATCH --time             01:00:00
#SBATCH --job-name         "xgc.rct"
#SBATCH --chdir            $RUN_DIR
#SBATCH --output           "slurm.rct.out"
#SBATCH --error            "slurm.rct.err"
#SBATCH --core-spec        8
#SBATCH --threads-per-core 1
#SBATCH --constraint       nvme
#SBATCH --export           NONE
#SBATCH --network          job_vni
#SBATCH --reservation      hackathon3

unset SLURM_EXPORT_ENV

# activate environment with RADICAL tools installed
# (could be a separate environment from the environment with scientific tools)
module load miniforge3
source activate /lustre/orion/world-shared/phy122/AI-Hackathon_2025/rhager/conda_xgc_pytorch

export RADICAL_PILOT_BASE=$RUN_DIR
export RADICAL_CONFIG_USER_DIR=$RUN_DIR

python $RUN_DIR/xgc-base.py -p $RP_PROJECT -n $N_NODES

EOT
