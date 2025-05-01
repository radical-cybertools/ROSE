import os
import sys
import time
import shutil

verbose  = os.environ.get('RADICAL_PILOT_VERBOSE', 'REPORT')
os.environ['RADICAL_PILOT_VERBOSE'] = verbose

import radical.pilot as rp
import radical.utils as ru

from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine

exe = f"{sys.executable}"

NNODES=1
nthread=32
nthread_tot=( NNODES * nthread )

#nthread_study=22
nthread_study=10
nrank_ml=4
ngpus_tot=(NNODES * 4)

MAX_ITER=4

engine = ResourceEngine({'resource': 'anl.polaris', 
                         'runtime' : 60,
                         'access_schema':'interactive',
                         'project' : "RECUP",
                         'cores'   : nthread_tot, 
                         'gpus'    : ngpus_tot}) 


#num_sample=4500
num_sample=2000
num_al_sample=((num_sample * 3))
batch_size=512
#epochs_list=[400,300,250,200]
epochs_list=[200,150,125,100]

work_dir="/lus/eagle/projects/RECUP/twang/exalearn_stage2/"
exe_dir=f"{work_dir}/executable/"

exp_dir_template="{work_dir}/experiment/seed_{{seed}}/".format(work_dir = work_dir)
shared_file_dir_template="{exp_dir_template}/sfd/".format(exp_dir_template = exp_dir_template)
data_dir_template="{work_dir}/data/seed_{{seed}}/".format(work_dir = work_dir)


acl = ActiveLearner(engine)
pre_exec_list = ["source /home/twang3/useful_script/conda_rose.sh", "export MPICH_GPU_SUPPORT_ENABLED=1"]

@acl.utility_task
def sample_simulation(*args, arg_list):
    print("In sample_simulation(), with arg_list =\n", arg_list)
    return Task(executable=f'{exe} {exe_dir}/simulation_sample.py {arg_list}', cores_per_rank=1, pre_exec=pre_exec_list, ranks=nthread)

@acl.utility_task
def sweep_simulation(*args, arg_list):
    print("In sweep_simulation(), with arg_list =\n", arg_list)
    return Task(executable=f'{exe} {exe_dir}/simulation_sweep.py {arg_list}', cores_per_rank=1, pre_exec=pre_exec_list, ranks=nthread_study)

@acl.utility_task
def merge_preprocess(*args, arg_list):
    print("In merge_preprocess(), with arg_list =\n", arg_list)
    return Task(executable=f'{exe} {exe_dir}/merge_preprocess_hdf5.py {arg_list}', cores_per_rank=2, pre_exec=pre_exec_list, threading_type = rp.OpenMP)

@acl.utility_task
def preprocess_study(*args, data_dir):
    print("In preprocess_study(), with data_dir = ", data_dir)
    return Task(executable=f'{exe} {exe_dir}/preprocess_study.py --data_dir {data_dir}', cores_per_rank=1, pre_exec=pre_exec_list)

@acl.training_task
def training(*args, arg_list):
    print("In training(), with arg_list =\n", arg_list)
    return Task(executable=f'{exe} {exe_dir}/train.py {arg_list}', cores_per_rank=8, gpus_per_rank=1, pre_exec=pre_exec_list+["export OMP_NUM_THREADS=8"], ranks=4)

@acl.active_learn_task
def active_learn(*args, arg_list):
    print("In active_learn(), with arg_list =\n", arg_list)
    return Task(executable=f'{exe} {exe_dir}/active_learning.py {arg_list}', cores_per_rank=32, pre_exec=pre_exec_list, threading_type = rp.OpenMP)

@acl.simulation_task
def simulation(*args, arg_list):
    print("In simulation(), with arg_list =\n", arg_list)
    return Task(executable=f'{exe} {exe_dir}/simulation_resample.py {arg_list}', cores_per_rank=1, pre_exec=pre_exec_list, ranks=nthread)



def remove_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"Removed directory {path!r}")
    else:
        print(f"No directory at {path!r}")

def bootstrap(seed):
    exp_dir = exp_dir_template.format(seed=seed)
    shared_file_dir = shared_file_dir_template.format(seed=seed)
    data_dir = data_dir_template.format(seed=seed)
    print(f"Logging: Start! seed = {seed}")
    print(f"Logging: \nexp_dir = {exp_dir}\nshared_file_dir = {shared_file_dir}\ndata_dir = {data_dir}")
    print("Logging: Doing cleaning")

    remove_dir(exp_dir)
    remove_dir(data_dir)
    os.makedirs(exp_dir, exist_ok=False)
    os.system(f'{exe} {work_dir}/prepare_data_dir.py --seed {seed}')

    args_template = '{num_sample} {{specific_seed}} {data_dir}/{{data_type}}/config/config_1001460_cubic.txt {data_dir}/{{data_type}}/config/config_1522004_trigonal.txt {data_dir}/{{data_type}}/config/config_1531431_tetragonal.txt'.format(num_sample=num_sample, data_dir=data_dir)
    
    bootstrap=[]
    base  = sample_simulation(arg_list=args_template.format(specific_seed=seed,   data_type="base"))
    test  = sample_simulation(arg_list=args_template.format(specific_seed=seed+1, data_type="test"))
    study = sweep_simulation(arg_list =args_template.format(specific_seed='',     data_type="study"))
    bootstrap.append(base)
    bootstrap.append(test)
    bootstrap.append(study)
    for shape in ['cubic', 'trigonal', 'tetragonal']:
        merge_base  = merge_preprocess(base,  arg_list=f'{data_dir}/base/data {shape} {nthread}')
        merge_test  = merge_preprocess(test,  arg_list=f'{data_dir}/test/data {shape} {nthread}')
        merge_study = merge_preprocess(study, arg_list=f'{data_dir}/study/data {shape} {nthread_study}')
        bootstrap.append(merge_base)
        bootstrap.append(merge_test)
        bootstrap.append(merge_study)
    
    [task.result() for task in bootstrap]
    study_preprocess = preprocess_study(data_dir=data_dir)
    study_preprocess.result()

def teach(seed):
    exp_dir = exp_dir_template.format(seed=seed)
    shared_file_dir = shared_file_dir_template.format(seed=seed)
    data_dir = data_dir_template.format(seed=seed)
    print(f"Logging: Start! seed = {seed}")
    print(f"Logging: \nexp_dir = {exp_dir}\nshared_file_dir = {shared_file_dir}\ndata_dir = {data_dir}")
    print("Logging: Doing cleaning")

    train_arg_list_template = "--batch_size {batch_size} --epochs {{epochs}} --seed {seed} --device=gpu --num_threads 8 --phase_idx {{iter_id}} --data_dir {data_dir} --shared_file_dir {shared_file_dir}".format(batch_size=batch_size, seed=seed, data_dir=data_dir, shared_file_dir=shared_file_dir)
    for iter_id in range(MAX_ITER):
        print(f'Starting Iteration-{iter_id}')
        remove_dir(shared_file_dir)
        os.makedirs(shared_file_dir, exist_ok=False)

        train_arg_list = train_arg_list_template.format(epochs=epochs_list[iter_id], iter_id=iter_id)
        train = training(arg_list=train_arg_list)
        train.result()

        if iter_id < MAX_ITER-1:
            active = active_learn(arg_list = f"--seed {seed+iter_id+1} --num_new_sample {num_al_sample} --policy uncertainty --data_dir {data_dir}")

            sim = simulation(active, arg_list=f'{seed+2+iter_id} {data_dir}/AL_phase_{iter_id+1}/config/config_1001460_cubic.txt {data_dir}/study/data/cubic_1001460_cubic.hdf5 {data_dir}/AL_phase_{iter_id+1}/config/config_1522004_trigonal.txt {data_dir}/study/data/trigonal_1522004_trigonal.hdf5 {data_dir}/AL_phase_{iter_id+1}/config/config_1531431_tetragonal.txt {data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5')
            sim.result()
            
            merge_task = []
            for shape in ['cubic', 'trigonal', 'tetragonal']:
                merge=merge_preprocess(arg_list=f'{data_dir}/AL_phase_{iter_id+1}/data {shape} {nthread}')
                merge_task.append(merge)
            [merge.result() for merge in merge_task]

bootstrap(seed=20071)
teach(seed=20071)
engine.shutdown()
