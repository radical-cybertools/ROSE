#!/bin/bash

seed=10010
work_dir="/lus/eagle/projects/RECUP/twang/exalearn_stage2/"
exe_dir="${work_dir}/executable/"
exp_dir="${work_dir}/experiment/seed_${seed}/"
shared_file_dir="${exp_dir}/sfd/"
data_dir="${work_dir}/data/seed_${seed}/"
num_sample=10000
num_al_sample=30000
nthread_all=32
nthread_part=24
nthread_small=10
batch_size=512
epochs=400

if [ -d "${exp_dir}" ]; then
    echo "Dir ${exp_dir} exist!"    
    exit 1
fi

mkdir -p ${exp_dir}
cd ${exp_dir}

{
    echo "Logging: Start! seed = ${seed}"
    echo "Logging: data_dir = ${data_dir}"
    echo "Logging: Doing cleaning"
    mv -f ${data_dir} /lus/eagle/projects/RECUP/twang/trashbin/
    python3 ${work_dir}/prepare_data_dir.py --seed ${seed}
 
    
    echo "Logging: Start base simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread_all} --ppn ${nthread_all} \
         python ${exe_dir}/simulation_sample.py \
                 ${num_sample} ${seed} \
                 ${data_dir}/base/config/config_1001460_cubic.txt \
                 ${data_dir}/base/config/config_1522004_trigonal.txt \
                 ${data_dir}/base/config/config_1531431_tetragonal.txt
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data cubic ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data trigonal ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data tetragonal ${nthread_all}

    echo "Logging: End base simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start test simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread_all} --ppn ${nthread_all} \
         python ${exe_dir}/simulation_sample.py \
                 ${num_sample} $((${seed} + 1)) \
                 ${data_dir}/test/config/config_1001460_cubic.txt \
                 ${data_dir}/test/config/config_1522004_trigonal.txt \
                 ${data_dir}/test/config/config_1531431_tetragonal.txt
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data cubic ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data trigonal ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data tetragonal ${nthread_all}
    echo "Logging: End test simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start study simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread_small} --ppn ${nthread_small} \
         python ${exe_dir}/simulation_sweep.py \
                 ${num_sample} \
                 ${data_dir}/study/config/config_1001460_cubic.txt \
                 ${data_dir}/study/config/config_1522004_trigonal.txt \
                 ${data_dir}/study/config/config_1531431_tetragonal.txt
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data cubic ${nthread_small}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data trigonal ${nthread_small}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data tetragonal ${nthread_small}

    echo "Logging: End study simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
   
    echo "Logging: Start training, phase 0"
    mv -f ${shared_file_dir} /lus/eagle/projects/RECUP/twang/trashbin/
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 0 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start stream simulation and merge!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread_all} --ppn ${nthread_all} \
         python ${exe_dir}/simulation_sample.py \
                 ${num_sample} $((${seed} + 1)) \
                 ${data_dir}/stream_phase_0/config/config_1001460_cubic.txt \
                 ${data_dir}/stream_phase_0/config/config_1522004_trigonal.txt \
                 ${data_dir}/stream_phase_0/config/config_1531431_tetragonal.txt
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_0/data cubic ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_0/data trigonal ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/stream_phase_0/data tetragonal ${nthread_all}
    echo "Logging: End stream simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start AL, phase 0"
    start=$(date +%s%3N)
    python ${exe_dir}/active_learning.py --seed $((${seed} + 1)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    for ((phase_idx=1; phase_idx <= num_phase - 1; phase_idx++));
    do
        echo "Logging: Start resample prepare, phase 1!"
        start=$(date +%s%3N)
        python ${exe_dir}/simulation_resample_prepare.py \
            $((${seed} + 2)) \
            ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
            ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
            ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5
        echo "Logging: End resample prepare, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
        echo "Logging: Start resample simulation and merge, first, phase 1!"
        start=$(date +%s%3N)
        mpiexec -n ${nthread_all} --ppn ${nthread_all} \
             python ${exe_dir}/simulation_resample_real_work.py \
                     $((${seed} + 2)) \
                     ${data_dir}/AL_phase_1/config/config_1001460_cubic.txt \
                     ${data_dir}/AL_phase_1/config/config_1522004_trigonal.txt \
                     ${data_dir}/AL_phase_1/config/config_1531431_tetragonal.txt \
                     "first" \
                     0.5
     
        python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data cubic ${nthread_all}
        python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data trigonal ${nthread_all}
        python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data tetragonal ${nthread_all}
        echo "Logging: End resample simulation and merge, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
        
        echo "Logging: Start training, phase 1"
        mv ${shared_file_dir} /lus/eagle/projects/RECUP/twang/trashbin/
        mkdir -p ${shared_file_dir}
        start=$(date +%s%3N)
        mpiexec -n 4 --ppn 8 \
            python ${exe_dir}/train.py --device=gpu --phase_idx=1 --data_dir=${data_dir} --blind_train_epoch 400 --shared_file_dir ${shared_file_dir}
        echo "Logging: End training, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
     
        echo "Logging: Start resample simulation and merge, second, phase 1!"
        start=$(date +%s%3N)
        mpiexec -n ${nthread_all} --ppn ${nthread_all} \
             python ${exe_dir}/simulation_resample_real_work.py \
                     $((${seed} + 2)) \
                     ${data_dir}/AL_phase_1/config/config_1001460_cubic.txt \
                     ${data_dir}/AL_phase_1/config/config_1522004_trigonal.txt \
                     ${data_dir}/AL_phase_1/config/config_1531431_tetragonal.txt \
                     "second" \
                     0.5
     
        python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data cubic ${nthread_all}
        python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data trigonal ${nthread_all}
        python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data tetragonal ${nthread_all}
        echo "Logging: End resample simulation and merge, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
     
        rm AL-freq.npy
     
        echo "Logging: Start AL, phase 1"
        start=$(date +%s%3N)
        python ${exe_dir}/active_learning.py --seed 42 --num_new_sample ${num_al_sample} --policy uncertainty
        echo "Logging: End AL phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
    done



    echo "Logging: Start resample prepare, phase 1!"
    start=$(date +%s%3N)
    python ${exe_dir}/simulation_resample_prepare.py \
        $((${seed} + 2)) \
        ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
        ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
        ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5
    echo "Logging: End resample prepare, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    echo "Logging: Start resample simulation and merge, first, phase 1!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread_all} --ppn ${nthread_all} \
         python ${exe_dir}/simulation_resample_real_work.py \
                 $((${seed} + 2)) \
                 ${data_dir}/AL_phase_1/config/config_1001460_cubic.txt \
                 ${data_dir}/AL_phase_1/config/config_1522004_trigonal.txt \
                 ${data_dir}/AL_phase_1/config/config_1531431_tetragonal.txt \
                 "first" \
                 0.5
    
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data cubic ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data trigonal ${nthread_all}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data tetragonal ${nthread_all}
    echo "Logging: End resample simulation and merge, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start training, phase 1"
    mv -f ${shared_file_dir} /lus/eagle/projects/RECUP/twang/trashbin/
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 1 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    rm AL-freq.npy
 
    echo "Logging: Start AL, phase 1"
    start=$(date +%s%3N)
    python ${exe_dir}/active_learning.py --seed $((${seed} + 2)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start resample simulation and merge, phase 2!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        python ${exe_dir}/simulation_resample.py \
                $((${seed} + 3)) \
                ${data_dir}/AL_phase_2/config/config_1001460_cubic.txt \
                ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
                ${data_dir}/AL_phase_2/config/config_1522004_trigonal.txt \
                ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
                ${data_dir}/AL_phase_2/config/config_1531431_tetragonal.txt \
                ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data cubic ${nthread}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data trigonal ${nthread}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    echo "Logging: Start training, phase 2"
    mv -f ${shared_file_dir} /lus/eagle/projects/RECUP/twang/trashbin/
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 2 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    rm AL-freq.npy

    echo "Logging: Start AL, phase 2"
    start=$(date +%s%3N)
    python ${exe_dir}/active_learning.py --seed $((${seed} + 3)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample simulation and merge, phase 3!"
    start=$(date +%s%3N)
    mpiexec -n ${nthread} --ppn ${nthread} \
        python ${exe_dir}/simulation_resample.py \
               $((${seed} + 4)) \
               ${data_dir}/AL_phase_3/config/config_1001460_cubic.txt \
               ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
               ${data_dir}/AL_phase_3/config/config_1522004_trigonal.txt \
               ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
               ${data_dir}/AL_phase_3/config/config_1531431_tetragonal.txt \
               ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5

    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data cubic ${nthread}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data trigonal ${nthread}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data tetragonal ${nthread}
    echo "Logging: End resample simulation and merge, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start training, phase 3"
    mv -f ${shared_file_dir} /lus/eagle/projects/RECUP/twang/trashbin/
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    mpiexec -n 4 --ppn 4 \
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 8 \
                                   --phase_idx 3 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"
   
    rm AL-freq.npy
 
    echo "Logging: All done for seed = ${seed}"
} > "log_${seed}_out" 2> "log_${seed}_err"
