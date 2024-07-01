#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q debug
#PBS -A RECUP

source /home/twang3/useful_script/conda_exalearn.sh
export MPICH_GPU_SUPPORT_ENABLED=1

seed=14000
work_dir="/lus/eagle/projects/RECUP/twang/exalearn_stage2/"
exe_dir="${work_dir}/executable/"
exp_dir="${work_dir}/experiment/seed_${seed}/"
shared_file_dir="${exp_dir}/sfd/"
data_dir="${work_dir}/data/seed_${seed}/"
num_sample=4500
num_al_sample=16200
num_al_sample_final_stage=13500
batch_size=512
epochs_0=400
epochs_1=300
epochs_2=250
epochs_3=200

nthread=32
nthread_study=22
 
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
               ${num_sample} $((${seed} + 1)) \
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
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_0} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 0 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start study simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread_study} --ppn ${nthread_study} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_sweep.py \
                ${num_sample} \
                ${data_dir}/study/config/config_1001460_cubic.txt \
                ${data_dir}/study/config/config_1522004_trigonal.txt \
                ${data_dir}/study/config/config_1531431_tetragonal.txt
 
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data cubic ${nthread_study}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data trigonal ${nthread_study}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data tetragonal ${nthread_study}
    echo "Logging: End study simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
   
    echo "Logging: Start preprocessing study set"
    start=$(date +%s%3N)
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
        python ${exe_dir}/preprocess_study.py --data_dir ${data_dir}
    echo "Logging: End preprocessing study set, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start AL, phase 0"
    start=$(date +%s%3N)
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
        python ${exe_dir}/active_learning.py --seed $((${seed} + 2)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample prepare, phase 1!"
    start=$(date +%s%3N)    
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_prepare.py \
               $((${seed} + 3)) \
               ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
               ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
               ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5
    echo "Logging: End resample prepare, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample simulation and merge, first (AL), phase 1!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_real_work.py \
               ${data_dir}/AL_phase_1/config/config_1001460_cubic.txt \
               ${data_dir}/AL_phase_1/config/config_1522004_trigonal.txt \
               ${data_dir}/AL_phase_1/config/config_1531431_tetragonal.txt \
               "first" \
               0.5
    
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, first (AL), phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start training, phase 1"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        --depth=8 --cpu-bind depth --env OMP_NUM_THREADS=8 --env OMP_PLACES=threads \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_1} \
                                   --seed $((${seed} + 1)) \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 1 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start resample simulation and merge, second (stream), phase 1!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_real_work.py \
               ${data_dir}/stream_phase_1/config/config_1001460_cubic.txt \
               ${data_dir}/stream_phase_1/config/config_1522004_trigonal.txt \
               ${data_dir}/stream_phase_1/config/config_1531431_tetragonal.txt \
               "second" \
               0.5

    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_1/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_1/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_1/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, second (stream), phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"

    rm AL-freq.npy
 
    echo "Logging: Start AL, phase 1"
    start=$(date +%s%3N)
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
        python ${exe_dir}/active_learning.py --seed $((${seed} + 4)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start resample prepare, phase 2!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_prepare.py \
               $((${seed} + 5)) \
               ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
               ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
               ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5 
    echo "Logging: End resample prepare, phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample simulation and merge, first (AL), phase 2!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_real_work.py \
               ${data_dir}/AL_phase_2/config/config_1001460_cubic.txt \
               ${data_dir}/AL_phase_2/config/config_1522004_trigonal.txt \
               ${data_dir}/AL_phase_2/config/config_1531431_tetragonal.txt \
               "first" \
               0.5

    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, first (AL), phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    echo "Logging: Start training, phase 2"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        --depth=8 --cpu-bind depth --env OMP_NUM_THREADS=8 --env OMP_PLACES=threads \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_2} \
                                   --seed $((${seed} + 2)) \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 2 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    echo "Logging: Start resample simulation and merge, second (stream), phase 2!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_real_work.py \
               ${data_dir}/stream_phase_2/config/config_1001460_cubic.txt \
               ${data_dir}/stream_phase_2/config/config_1522004_trigonal.txt \
               ${data_dir}/stream_phase_2/config/config_1531431_tetragonal.txt \
               "second" \
               0.5

    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_2/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_2/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_2/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, second (stream), phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"

    rm AL-freq.npy

    echo "Logging: Start AL, phase 2"
    start=$(date +%s%3N)
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads \
        python ${exe_dir}/active_learning.py --seed $((${seed} + 6)) --num_new_sample ${num_al_sample_final_stage} --policy uncertainty
    echo "Logging: End AL phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample prepare, phase 3!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_prepare.py \
               $((${seed} + 7)) \
               ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
               ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
               ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5
    echo "Logging: End resample prepare, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample simulation and merge, first (AL), phase 3!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        --depth=1 --cpu-bind depth --env OMP_NUM_THREADS=1 --env OMP_PLACES=threads \
        python ${exe_dir}/simulation_resample_real_work.py \
               ${data_dir}/AL_phase_3/config/config_1001460_cubic.txt \
               ${data_dir}/AL_phase_3/config/config_1522004_trigonal.txt \
               ${data_dir}/AL_phase_3/config/config_1531431_tetragonal.txt \
               "first" \
               1.0

    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data cubic ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data trigonal ${nthread}
    mpiexec -n 1 --ppn 1 --depth=32 --cpu-bind depth --env OMP_NUM_THREADS=32 --env OMP_PLACES=threads python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start training, phase 3"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        --depth=8 --cpu-bind depth --env OMP_NUM_THREADS=8 --env OMP_PLACES=threads \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_3} \
                                   --seed $((${seed} + 3)) \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 3 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"
   
    rm AL-freq.npy
 
    echo "Logging: All done for seed = ${seed}"
}
