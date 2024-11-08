from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru
import json
import math
import time

class DiffusionSolver(object):

    def __init__(self):
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
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


        args = parser.parse_args()
        self.args = args

#use a trivial sleep function in the place where we need a simulation as a temp solution
    def run_sim(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()
        
        t.executable = '/bin/sleep'
        t.arguments = [self.args.sim_time]
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 1,
            'cpu_thread_type'   : rp.OpenMP
            }
        s.add_tasks(t)

        return s

    def run_train(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()
        
        initial_task = 1 if phase_idx == 0 else 0
        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.src_dir}/ml_and_al/train_net.py",
            '--card=0',
            f'--initial={initial_task}',
            f'--portion={phase_idx+1}',
            f'--epochs={self.args.epochs}',
            '--es=1',
            ]
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
            }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
            }
        s.add_tasks(t)

        return s

    def run_al(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()
        
        initial_task = 1 if phase_idx == 0 else 0
        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.src_dir}/ml_and_al/active.py",
            '--card=0',
            f'--portion={phase_idx+1}',
            f'--func={self.args.al_func}',
            ]
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 8,
            'cpu_thread_type'   : rp.OpenMP
            }
        t.gpu_reqs = {
            'gpu_processes'     : 1,
            'gpu_process_type'  : rp.CUDA
            }
        s.add_tasks(t)

        return s

    def generate_pipeline(self):
        p = entk.Pipeline()
        for phase in range(int(self.args.num_phases)):
            s1 = self.run_sim(phase)
            p.add_stages(s1)
            s2 = self.run_train(phase)
            p.add_stages(s2)
            s3 = self.run_al(phase)
            p.add_stages(s3)
        return p

    def run_workflow(self):
        p = self.generate_pipeline()
        self.am.workflow = [p]
        self.am.run()


if __name__ == "__main__":
    wf = DiffusionSolver()
    wf.set_resource(res_desc = {
        'resource': 'anl.polaris',
        'queue'   : wf.args.queue,
        'walltime': 60, #MIN
        'cpus'    : 32 * wf.args.num_nodes,
        'gpus'    : 4 * wf.args.num_nodes,
        'project' : wf.args.project_id
        })
    wf.run_workflow()
