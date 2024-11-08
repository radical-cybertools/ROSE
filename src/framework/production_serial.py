from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru
import json
import math
import time

class SurrogateTraining(object):

    def __init__(self):
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="Surrogate_training_using_rose_in_production_mode")

        parser.add_argument('--num_phases', type=int, required=True,
                            help='number of phases for doing active learning')
        parser.add_argument('--sim_script',
                            help='where simulation script is located. Assumed to be a python script.')
        parser.add_argument('--sim_config',
                            help='the config file for simulation script')
        parser.add_argument('--train_script',
                            help='where training script is located. Assumed to be a python script.')
        parser.add_argument('--train_config',
                            help='the config file for training script')
        parser.add_argument('--al_func', required=True,
                            help='the aciquision function used in active learning policy')
        parser.add_argument('--conda_env', default=None, required=True,
                            help='the conda env where dependency is installed')
        parser.add_argument('--project_id', required=True,
                            help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                            help='the queue we used to submit the job')
        parser.add_argument('--num_nodes', type=int, default=1,
                            help='number of nodes used for the workflow')
        parser.add_argument('--seed', type=int, default=2024,
                            help='the root seed used for this project')
        parser.add_argument('--num_reprod', type=int, default=2,
                            help='number of experiment done with different seed')

        args = parser.parse_args()
        self.args = args

#This function return a seed for each task (sim, train, al) at each phase with a basic_seed that is different for each pipeline
    def get_seed(self, basic_seed, task_type, phase):
        if task_type == 'sim':
            return basic_seed * 1001 + phase * 11 + 1
        elif task_type == 'train':
            return basic_seed * 1001 + phase * 11 + 2
        elif task_type == 'al':
            return basic_seed * 1001 + phase * 11 + 3
        else:
            raise ValueError(f'task_type is not set up correctly: {task_type}')

    def run_sim(self, phase_idx, basic_seed):
        s = entk.Stage()
        t = entk.Task()
        seed = self.get_seed(basic_seed, "sim", phase_idx)

        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.sim_script}",
            f'--config={self.args.sim_config}',
            f'--seed={seed}',
            ]
        t.cpu_reqs = {
            'cpu_processes'     : 1,
            'cpu_process_type'  : None,
            'cpu_threads'       : 1,
            'cpu_thread_type'   : rp.OpenMP
            }
        s.add_tasks(t)

        return s

    def run_train(self, phase_idx, basic_seed):
        s = entk.Stage()
        t = entk.Task()
        seed = self.get_seed(basic_seed, "train", phase_idx)
        
        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.train_script}",
            f'--config={self.args.train_config}',
            f'--seed={seed}',
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

    def run_al(self, phase_idx, basic_seed):
        s = entk.Stage()
        t = entk.Task()
        seed = self.get_seed(basic_seed, "al", phase_idx)
        
        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.al_func}.py",
            f'--seed={seed}',
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

    def generate_pipeline(self, basic_seed):
        p = entk.Pipeline()
        for phase in range(int(self.args.num_phases)):
            s1 = self.run_sim(phase, basic_seed)
            p.add_stages(s1)
            if phase != int(self.args.num_phases)-1:
                s2 = self.run_train(phase, basic_seed)
                p.add_stages(s2)
                s3 = self.run_al(phase, basic_seed)
                p.add_stages(s3)
        return p

    def run_workflow(self):
        pipeline_list = []
        for basic_seed in range(self.args.seed, self.args.seed + self.args.num_reprod):
            p = self.generate_pipeline(basic_seed)
            pipeline_list.append(p)
        self.am.workflow = pipeline_list
        self.am.run()


if __name__ == "__main__":
    wf = SurrogateTraining()
    wf.set_resource(res_desc = {
        'resource': 'anl.polaris',
        'queue'   : wf.args.queue,
        'walltime': 60, #MIN
        'cpus'    : 32 * wf.args.num_nodes,
        'gpus'    : 4 * wf.args.num_nodes,
        'project' : wf.args.project_id
        })
