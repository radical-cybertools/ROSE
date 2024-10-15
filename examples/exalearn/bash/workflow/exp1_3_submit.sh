#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug
#PBS -A RECUP

source /home/twang3/useful_script/conda_exalearn.sh
export MPICH_GPU_SUPPORT_ENABLED=1

seed=12380
work_dir="/lus/eagle/projects/RECUP/twang/exalearn_stage2/"
exe_dir="${work_dir}/executable/"
exp_dir="${work_dir}/experiment/seed_${seed}/"
shared_file_dir="${exp_dir}/sfd/"
data_dir="${work_dir}/data/seed_${seed}/"
num_sample=72000
num_sample_test=$((${num_sample} / 4))
batch_size=512
epochs_0=400

nthread=32
echo "Logging: Start! seed = ${seed}"
echo "Logging: data_dir = ${data_dir}"
echo "Logging: Doing cleaning"
rm -r ${exp_dir}
rm -r ${data_dir}

mkdir -p ${exp_dir}
cd ${exp_dir}

################################  Start real job  ##############################

{
    set -e

    mpiexec -n 1 --ppn 1 \
        --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
        python3 ${work_dir}/prepare_data_dir.py --seed ${seed}
    
    echo "Logging: Start base simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_sample.py \
               ${num_sample} ${seed} \
               ${data_dir}/base/config/config_1001460_cubic.txt \
               ${data_dir}/base/config/config_1522004_trigonal.txt \
               ${data_dir}/base/config/config_1531431_tetragonal.txt

    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data tetragonal ${nthread}
    echo "Logging: End base simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start test simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_sample.py \
               ${num_sample_test} $((${seed} + 1)) \
               ${data_dir}/test/config/config_1001460_cubic.txt \
               ${data_dir}/test/config/config_1522004_trigonal.txt \
               ${data_dir}/test/config/config_1531431_tetragonal.txt

    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data tetragonal ${nthread}
    echo "Logging: End test simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start training, phase 0"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        --depth=8 --cpu-bind depth --env OMP_NUM_THREADS=8 --env OMP_PLACES=threads \
        python ${exe_dir}/train_without_study.py --batch_size ${batch_size} \
                                   --epochs ${epochs_0} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 0 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: All done for seed = ${seed}"
}
