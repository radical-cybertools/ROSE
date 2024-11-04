import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru
import time

module_dir = "/eagle/RECUP/twang/rose/rose_github/src/framework/"
sys.path.append(module_dir)

from data import InputFile, OutputFile
from roseee import ResourceEngine, WorkflowEngine
from rose import SimulationTask, ActiveLearnTask, TrainingTask

def set_argparse():
    parser = argparse.ArgumentParser(description="Diffusion_Solver_entk_for_rose")

    parser.add_argument('--num_phases', type=int, default=3,
                        help='number of phases for doing active learning')
    parser.add_argument('--sim_time', type=int, default=10,
                        help='the fake simulation task is a sleep function, how long in seconds it sleeps')
    parser.add_argument('--src_dir', default=None, required=True,
                        help='the source directory of sim/ml/al tasks')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs in training task')
    parser.add_argument('--al_func', choices=['tod', 'random'], required=True,
                        help='the aciquision function used in active learning policy')
    parser.add_argument('--conda_env', default=None, required=True,
                        help='the conda env where dependency is installed')
    parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
    parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
    parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes used for the workflow')

    my_args = parser.parse_args()
    return my_args


my_args = set_argparse()
res_desc = {
        'resource': 'anl.polaris',
        'queue'   : my_args.queue,
        'walltime': 60, #MIN
        'cpus'    : 32 * my_args.num_nodes,
        'gpus'    : 4 * my_args.num_nodes,
        'project' : my_args.project_id
        }

engine = ResourceEngine(res_desc)
flow = WorkflowEngine(engine=engine)

@flow
def sim(*args, my_args, phase_idx):
    cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 1,
            'cpu_thread_type'   : rp.OpenMP
            }
    return SimulationTask(executable='/bin/sleep',
                          arguments=[my_args.sim_time],
                          **cpu_reqs)

@flow
def train(*args, my_args, phase_idx):
    initial_task = 1 if phase_idx == 0 else 0
    pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {my_args.conda_env}",
            ]
    arguments = [
            f"{my_args.src_dir}/ml_and_al/train_net.py",
            '--card=0',
            f'--initial={initial_task}',
            f'--portion={phase_idx+1}',
            f'--epochs={my_args.epochs}',
            '--es=1',
            ]
    cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
            }
    gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
            }
    return TrainingTask(executable='python',
                        pre_exec=pre_exec,
                        arguments=arguments,
                        **cpu_reqs,
                        **gpu_reqs)


@flow
def acl(*args, my_args, phase_idx):
    initial_task = 1 if phase_idx == 0 else 0
    pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {my_args.conda_env}",
            ]
    arguments = [
            f"{my_args.src_dir}/ml_and_al/active.py",
            '--card=0',
            f'--portion={phase_idx+1}',
            f'--func={my_args.al_func}',
            ]
    cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
            }
    gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
            }
    return ActiveLearnTask(executable='python',
                           pre_exec=pre_exec,
                           arguments=arguments,
                           **cpu_reqs,
                           **gpu_reqs)

sim_task = sim(my_args=my_args, phase_idx=0)
train_task = train(sim_task, my_args=my_args, phase_idx=0)
for phase in range(my_args.num_phases-1):
    acl_task = acl(train_task, my_args=my_args, phase_idx=phase)
    sim_task = sim(acl_task, my_args=my_args, phase_idx=phase+1)
    train_task = train(sim_task, my_args=my_args, phase_idx=phase+1)

flow.run()
engine.shutdown()
