from radical import entk
import os
import argparse, sys, math
import radical.pilot as rp
import radical.utils as ru
import json
import math
import time


class ActiveLearningAlgoSeletcion(object):

    def __init__(self):
        self.set_argparse()
        self.am = entk.AppManager()

    def set_resource(self, res_desc):
        self.am.resource_desc = res_desc

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="Active_learning_algorithm_selection_in_surrogate_training")

        parser.add_argument('--sim_script',
                            help='where simulation script is located. Assumed to be a python script.')
        parser.add_argument('--sim_config',
                            help='the config file for simulation script')
        parser.add_argument('--train_script',
                            help='where training script is located. Assumed to be a python script.')
        parser.add_argument('--train_config',
                            help='the config file for training script')
        parser.add_argument('--al_func_list', nargs='+', default=['random'],
                            help='the list of aciquision function being test')
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

        args = parser.parse_args()
        self.args = args

    def run_sim(self, phase_idx):
        s = entk.Stage()
        t = entk.Task()

        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.sim_script}",
            f'--config={self.args.sim_config}',
            f'--seed={self.args.seed}',
            ]
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
        
        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.train_script}",
            f'--config={self.args.train_config}',
            f'--seed={self.args.seed}',
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
        
        t.pre_exec = [
            "module use /soft/modulefiles",
            "module load conda",
            f"conda activate {self.args.conda_env}",
            ]
        t.executable = 'python'
        t.arguments = [
            f"{self.args.al_func}.py",
            f'--seed={self.args.seed}',
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

    def generate_pipeline(self, al_func):
        p = entk.Pipeline()
        s1 = self.run_sim(0)
        p.add_stages(s1)
        s2 = self.run_train(0)
        p.add_stages(s2)

        s3 = self.run_al(0, al_func)
        p.add_stages(s3)
        s4 = self.run_sim(1)
        p.add_stages(s4)
        s5 = self.run_train(1)
        p.add_stages(s5)

        return p

    def run_workflow(self):
        pipeline_list = []
        for al_func in self.args.al_func_list
            p = self.generate_pipeline(al_func)
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
