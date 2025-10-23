import os
import sys
import time

verbose  = os.environ.get('RADICAL_PILOT_VERBOSE', 'REPORT')
os.environ['RADICAL_PILOT_VERBOSE'] = verbose

from rose.learner import Learner

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

try:
    import numpy, sklearn
except ImportError:
    print("\nRun 'pip install numpy scikit-learn' to use this example.\n")
    sys.exit(1)

seed=20030
num_sample=4500
num_sample_val=((num_sample / 2))
num_sample_test=((num_sample / 2))
num_sample_study=num_sample
num_al_sample=((num_sample * 3))
batch_size=512
epochs=[400,300,250,200]

NNODES=1

nthread=32
nthread_tot=( NNODES * nthread )

nthread_study=22
nthread_study_tot=( NNODES * nthread_study )

nrank_ml=4
nrank_ml_tot=( NNODES * nrank_ml )

ngpus=(NNODES * 4)


async def bootstrap():
    os.system(f'{code_path}/prepare_data_dir_pm.py --seed {seed}')

    bootstrap=[]
    base = sample_simulation(f'{num_sample} {seed} \
            {data_dir}/base/config/config_1001460_cubic.txt \
            {data_dir}/base/config/config_1522004_trigonal.txt \
            {data_dir}/base/config/config_1531431_tetragonal.txt')
    val = sample_simulation(f'{num_sample} {seed-1} \
            {data_dir}/validation/config/config_1001460_cubic.txt \
            {data_dir}/validation/config/config_1522004_trigonal.txt \
            {data_dir}/validation/config/config_1531431_tetragonal.txt')
    test = sample_simulation(f'{num_sample} {seed+1} \
            {data_dir}/test/config/config_1001460_cubic.txt \
            {data_dir}/test/config/config_1522004_trigonal.txt \
            {data_dir}/test/config/config_1531431_tetragonal.txt')
    study = sweep_simulation(f'{num_sample_study} \
            {data_dir}/study/config/config_1001460_cubic.txt \
            {data_dir}/study/config/config_1522004_trigonal.txt \
            {data_dir}/study/config/config_1531431_tetragonal.txt')
    bootstrap.append(base)
    bootstrap.append(val)
    bootstrap.append(test)
    bootstrap.append(study)
    for shape in ['cubic', 'trigonal', 'tetragonal']:
        merge_base = merge_preprocess(f'{data_dir}/base/data {shape} {nthread_tot}', base)
        merge_val = merge_preprocess(f'{data_dir}/validation/data {shape} {nthread_tot}', val)
        merge_test = merge_preprocess(f'{data_dir}/test/data {shape} {nthread_tot}', test)
        merge_study = merge_preprocess(f'{data_dir}/study/data {shape} {nthread_tot}', study)
        bootstrap.append(merge_base)
        bootstrap.append(merge_val)
        bootstrap.append(merge_test)
        bootstrap.append(merge_study)

    [task.result() for task in bootstrap]


async def main():

    engine = await RadicalExecutionBackend(
        {'resource': 'local.localhost'})

    asyncflow = await WorkflowEngine.create(engine)
    acl = Learner(asyncflow)

    code_path = f'{sys.executable} {os.getcwd()}'

    data_dir= f'{os.getcwd()}/data/seed_{seed}'

    # The scripts used here are a dummy representative of the actual use case tasks, and
    #  dont actually run the simulation sample

    # Define and register the simulation task
    @acl.simulation_task
    async def simulation(*args):
        #return f'{code_path}/simulation_resample.py'
        return f'{code_path}/replacement_sim.py'

    # Define and register a utility task
    @acl.utility_task
    async def merge_preprocess(*args):
        # return f'{code_path}/merge_preprocess_hdf5.py'
        return f'{code_path}/replacement_sim.py'

    # Define and register the training task
    @acl.training_task
    async def training(*args):
        # return f'{code_path}/train.py'
        return f'{code_path}/replacement_sim.py'

    # Define and register the active learning task
    @acl.active_learn_task
    async def active_learn(*args):
        # return f'{code_path}/active_learning.py'
        return f'{code_path}/replacement_sim.py'

    # Prepare Data
    # Define the simulation sample task
    @acl.utility_task
    async def sample_simulation(*args):
        # task = f'{code_path}/simulation_sample.py'
        return f'{code_path}/replacement_sim.py'

    #simulation sweep task
    @acl.utility_task
    async def sweep_simulation(*args):
    #    task = f'{code_path}/simulation_sweep.py'
        return f'{code_path}/replacement_sim.py'

    # Custom training loop using active learning
    async def teach():
        for acl_iter in range(4):
            print(f'Starting Iteration-{acl_iter}')
            simulations = []
            if acl_iter != 0:
                sim = simulation(f'{seed+2} \
                    {data_dir}/AL_phase_{acl_iter}/config/config_1001460_cubic.txt \
                    {data_dir}/study/data/cubic_1001460_cubic.hdf5 \
                    {data_dir}/AL_phase_{acl_iter}/config/config_1522004_trigonal.txt \
                    {data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
                    {data_dir}/AL_phase_{acl_iter}/config/config_1531431_tetragonal.txt \
                    {data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5')
                simulations.append(sim)
                for shape in ['cubic', 'trigonal', 'tetragonal']:
                    merge=merge_preprocess(f'{data_dir}/AL_phase_{acl_iter}/data cubic {nthread_tot}', sim)
                    simulations.append(merge)

            await asyncio.gather(*simulations)
            # Now run training and active_learn
            train = training(f'--batch_size {batch_size} \
                --epochs {epochs[acl_iter]} \
                --seed {seed} \
                --device=cpu \
                --num_threads {nthread} \
                --phase_idx {acl_iter} \
                --data_dir {data_dir} \
                --shared_file_dir {data_dir}', *simulations)
            active = active_learn(f'--seed {seed+3} --num_new_sample {num_al_sample} --policy uncertainty', simulations, train)
            await active
    # invoke the custom/user-defined teach() method
    await bootstrap()
    await teach()
    await acl.shutdown()
