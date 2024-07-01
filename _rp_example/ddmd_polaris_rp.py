from radical import entk
import os
import argparse, sys, math
import traceback
import radical.pilot as rp
import radical.utils as ru
import json
import math
import threading as mt
import time


class MVP(object):
    TASK_MD             = 'task_md'
    TASK_TRAIN          = 'task_train'
    TASK_SELECT         = 'task_select'
    TASK_AGENT          = 'task_agent'

    TASK_TYPES = [TASK_MD, TASK_TRAIN, TASK_SELECT, TASK_AGENT]

    cores_used  = 0
    gpus_used   = 0
    avail_cores = 0
    avail_gpus  = 0

    def __init__(self):
        self.env_work_dir = os.getenv("MINI_APP_DeepDriveMD_DIR")
        if self.env_work_dir is None:
            print("Warning: Did not set up work_dir using env var, need to set it up in parser manually!")
#FIXME: This termination does not work, why???            
            sys.exit(1)
        self.stage_idx = 0

        self.set_argparse()
        self.get_json()

        self._protocol = {self.TASK_MD          : self._control_md,
                          self.TASK_TRAIN       : self._control_train,
                          self.TASK_SELECT      : self._control_select,
                          self.TASK_AGENT       : self._control_agent,
        }

        self._glyphs =   {self.TASK_MD          : 'm',
                          self.TASK_TRAIN       : 't',
                          self.TASK_SELECT      : 's',
                          self.TASK_AGENT       : 'a',
        }

        self._cores = 32
        self._gpus  = 4
        self._avail_gpus = self._gpus
        self._avail_cores = self._cores
        self._cores_used = 0
        self._gpus_used = 0

        self._lock = mt.RLock()
        self._uids  = list()
        self._tasks = {ttype: dict() for ttype in self.TASK_TYPES}
        self._final_tasks = list()

        os.environ['RADICAL_REPORT'] = 'false'
        self._rep = ru.Reporter('ddmd')
        self._rep.title('ddmd')

        self._session = rp.Session()
        self._pmgr = rp.PilotManager(session=self._session)
        self._tmgr = rp.TaskManager(session=self._session)

        pdesc = rp.PilotDescription({'resource': 'anl.polaris',
                                     'runtime' : 60,
                                     'queue'   : self.args.queue,
                                     'cores'   : self._cores * self.args.num_nodes,
                                     'gpus'    : self._gpus * self.args.num_nodes,
                                     'project' : self.args.project_id})
        self._pilot = self._pmgr.submit_pilots(pdesc)
        
        self._tmgr.add_pilots(self._pilot)
        self._tmgr.register_callback(self._state_cb)

    def __del__(self):
        self.close()

    def close(self):
        if self._session is not None:
            self._session.close(download=True)
            self._session = None

#FIXME: dump assumes one core per task, which is not true in many cases
    def dump(self, task=None, msg=''):
        # this assumes one core per task

        self._rep.plain('<<|')

        idle = self._cores

        for ttype in self.TASK_TYPES:

            n = 0
            n += len(self._tasks[ttype])
            idle -= n

            if n > self._cores:
                idle = 0
                n = self._cores

            self._rep.ok('%s' % self._glyphs[ttype] * n)

        self._rep.plain('%s' % '-' * idle +
                        '| %4d [%4d]' % (self._cores_used, self._cores))

        if task and msg:
            self._rep.plain(' %-15s: %s\n' % (task.uid, msg))
        else:
            if task:
                msg = task
            self._rep.plain(' %-15s: %s\n' % (' ', msg))

    def start(self):
        self.dump('submit MD simulations')
        time.sleep(1)
        print("Next is to self.generate_molecular_dynamics_stage")
        self.generate_molecular_dynamics_stage()
#FIXME: What should be the parameters?        
#        self._submit_task(self.TASK_MD, args=None, n=12, cpu=1, gpu=0, argvals='')

    def stop(self):
        os.kill(os.getpid(), signal.SIGKILL)
        os.kill(os.getpid(), signal.SIGTERM)

    def _get_ttype(self, uid):
        ttype = uid.split('.')[0]
        assert ttype in self.TASK_TYPES, 'unknown task type: %s' % uid
        return ttype

    def _submit_task(self, ttype):
        assert ttype
        if isinstance(ttype, list) and isinstance(ttype[0], rp.TaskDescription):
            tds = ttype
        elif isinstance(ttype, rp.TaskDescription):
            tds = [ttype]
        else:
            raise TypeError("invalid task type {}".format(type(ttype)))

        with self._lock:
            tasks = self._tmgr.submit_tasks(tds)
            for task in tasks:
                self._register_task(task)

    def _cancel_tasks(self, uids):
        uids = ru.as_list(uids)
        self._tmgr.cancel_tasks(uids)

        for uid in uids:
            ttype = self._get_ttype(uid)
            task  = self._tasks[ttype][uid]
            self.dump(task, 'cancel [%s]' % task.state)
            self._unregister_task(task)

        self.dump('cancelled')

    def _register_task(self, task):
        with self._lock:
            ttype = self._get_ttype(task.uid)
            self._uids.append(task.uid)
            self._tasks[ttype][task.uid] = task

            cores = task.description['ranks'] * task.description['cores_per_rank']
            self._cores_used += cores
            gpus = task.description['gpu_processes']
            self._gpus_used += gpus

    def _unregister_task(self, task):
        with self._lock:
            ttype = self._get_ttype(task.uid)
            if task.uid not in self._tasks[ttype]:
                return

            # remove task from bookkeeping
            self._final_tasks.append(task.uid)
            del self._tasks[ttype][task.uid]
            self.dump(task, 'unregister %s' % task.uid)

            cores = task.description['ranks'] * task.description['cores_per_rank']
            self._cores_used -= cores
            gpus = task.description['gpu_processes']
            self._gpus_used -= gpus

    def _state_cb(self, task, state):
        try:
            return self._check_state_cb(task, state)
        except Exception as e:
            self._rep.exception('\n\n----------\nexception caught: {}\n\n'.format(repr(e)))
            ru.print_exception_trace()
            self.stop()

    def _checked_state_cb(self, task, state):
  
        # this cb will react on task state changes.  Specifically it will watch
        # out for task completion notification and react on them, depending on
        # the task type.

        if state in [rp.TMGR_SCHEDULING] + rp.FINAL:
            self.dump(task, ' -> %s' % task.state)

        # ignore all non-final state transitions
        if state not in rp.FINAL:
            return

        # ignore tasks which were already completed
        if task.uid in self._final_tasks:
            return

        # lock bookkeeping
        with self._lock:

            # raise alarm on failing tasks (but continue anyway)
            if state == rp.FAILED:
                self._rep.error('task %s failed: %s' % (task.uid, task.stderr))
                self.stop()

            # control flow depends on ttype
            ttype  = self._get_ttype(task.uid)
            action = self._protocol[ttype]
            if not action:
                self._rep.exit('no action found for task %s' % task.uid)
            action(task)

            # remove final task from bookkeeping
            self._unregister_task(task)

    def _control_md(self, task):
        if len(self._tasks[self.TASK_MD]) > 1:
            return
        self.dump(task, 'completed MD')
        self.generate_machine_learning_stage()

    def _control_train(self, task):
        if len(self._tasks[self.TASK_TRAIN]) > 1:
            return
        self.dump(task, 'completed Train')
        self.generate_model_selection_stage()
    
    def _control_select(self, task):
        if len(self._tasks[self.TASK_SELECT]) > 1:
            return
        self.dump(task, 'completed Select')
        self.generate_agent_stage()

    def _control_agent(self, task):
        if len(self._tasks[self.TASK_DDMD_AGENT]) > 1:
            return
        self.dump(task, 'completed Agent')
        if self.stage_idx < self.args.num_phases:
            self.stage_idx += 1
            self.generate_molecular_dynamics_stage()
        else:
            self.dump("DONE!!!")
            ddmd.close()    #FIXME! What is this?

    def set_argparse(self):
        parser = argparse.ArgumentParser(description="DeepDriveMD_miniapp_RP_serial")

        parser.add_argument('--num_phases', type=int, default=3,
                        help='number of phases in the workflow')
        parser.add_argument('--mat_size', type=int, default=5000,
                        help='the matrix with have size of mat_size * mat_size')
        parser.add_argument('--data_root_dir', default='./',
                        help='the root dir of gsas output data')
        parser.add_argument('--num_step', type=int, default=1000,
                        help='number of step in MD simulation')
        parser.add_argument('--num_epochs_train', type=int, default=150,
                        help='number of epochs in training task')
        parser.add_argument('--model_dir', default='./',
                        help='the directory where save and load model')
        parser.add_argument('--conda_env', default=None,
                        help='the conda env where numpy/cupy installed, if not specified, no env will be loaded')
        parser.add_argument('--num_sample', type=int, default=500,
                        help='num of samples in matrix mult (training and agent)')
        parser.add_argument('--num_mult_train', type=int, default=4000,
                        help='number of matrix mult to perform in training task')
        parser.add_argument('--dense_dim_in', type=int, default=12544,
                        help='dim for most heavy dense layer, input')
        parser.add_argument('--dense_dim_out', type=int, default=128,
                        help='dim for most heavy dense layer, output')
        parser.add_argument('--preprocess_time_train', type=float, default=20.0,
                        help='time for doing preprocess in training')
        parser.add_argument('--preprocess_time_agent', type=float, default=10.0,
                        help='time for doing preprocess in agent')
        parser.add_argument('--num_epochs_agent', type=int, default=10,
                        help='number of epochs in agent task')
        parser.add_argument('--num_mult_agent', type=int, default=4000,
                        help='number of matrix mult to perform in agent task, inference')
        parser.add_argument('--num_mult_outlier', type=int, default=10,
                        help='number of matrix mult to perform in agent task, outlier')
        parser.add_argument('--enable_darshan', action='store_true',
                        help='enable darshan analyze')
        parser.add_argument('--project_id', required=True,
                        help='the project ID we used to launch the job')
        parser.add_argument('--queue', required=True,
                        help='the queue we used to submit the job')
        parser.add_argument('--work_dir', default=self.env_work_dir,
                        help='working dir, which is the dir of this repo')
        parser.add_argument('--num_sim', type=int, default=4,
                        help='number of tasks used for simulation')
        parser.add_argument('--num_nodes', type=int, default=1,
                        help='number of nodes used for simulation')
        parser.add_argument('--io_json_file', default="io_size.json",
                        help='the filename of json file for io size')

        args = parser.parse_args()
        self.args = args

    def get_json(self):
        json_file = "{}/launch-scripts/{}".format(self.args.work_dir, self.args.io_json_file)
        with open(json_file) as f:
            self.io_dict = json.load(f)

    def generate_task_description(self, td_dict):
        td = rp.TaskDescription()
        td.ranks            = td_dict["cpu_processes"]
        td.cores_per_rank   = td_dict["threads_per_process"]
        td.gpus_per_rank    = td_dict["gpu_processes"]
        td.pre_exec         = td_dict["pre_exec"]
        td.executable       = td_dict["executable"]
        td.arguments        = td_dict["arguments"]
        return td

    def generate_molecular_dynamics_stage(self):
        tds = []
        for i in range(self.args.num_sim):
            td = rp.TaskDescription()
            td.pre_exec = [
                    "module load PrgEnv-gnu",
                    "export HDF5_USE_FILE_LOCKING=FALSE",
                    "module use /soft/modulefiles",
                    "module load conda",
                    ]
            if self.args.conda_env is not None:
                td.pre_exec.append("conda activate {}".format(self.args.conda_env))
            td.pre_exec.append("which python")
            td.pre_exec.append("python -V")
            if self.args.enable_darshan:
                td.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
            else:
                td.executable = 'python'
            td.arguments = ['{}/Executables/simulation.py'.format(self.args.work_dir),
                           '--phase={}'.format(self.stage_idx),
                           '--task_idx={}'.format(i),
                           '--mat_size={}'.format(self.args.mat_size),
                           '--data_root_dir={}'.format(self.args.data_root_dir),
                           '--num_step={}'.format(self.args.num_step),
                           '--write_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["sim"]["write"]),
                           '--read_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["sim"]["read"])]
            td.post_exec = []
            td.ranks = 1
            td.cores_per_rank = 8
            td.gpus_per_rank = 1
            td.uid = ru.generate_id(self.TASK_MD)
            tds.append(td)
        self._submit_task(tds)

    def generate_machine_learning_stage(self):
        td = rp.TaskDescription()
        td.pre_exec = [
                "module load PrgEnv-gnu",
                "export HDF5_USE_FILE_LOCKING=FALSE",
                "module use /soft/modulefiles",
                "module load conda"
                ]
        if self.args.conda_env is not None:
            td.pre_exec.append("conda activate {}".format(self.args.conda_env))
        if self.args.enable_darshan:
            td.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        else:
            td.executable = 'python'
        td.arguments = ['{}/Executables/training.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_train),
                       '--device=gpu',
                       '--phase={}'.format(self.stage_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample * (1 if self.stage_idx == 0 else 2)),
                       '--num_mult={}'.format(self.args.num_mult_train),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_train),
                       '--write_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["train"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["train"]["read"])]
        td.post_exec = []
        td.ranks = 1
        td.cores_per_rank = 8
        td.gpus_per_rank = 1
        td.uid = ru.generate_id(self.TASK_TRAIN)
        self._submit_task(td)
#FIXME: Do we need gpu_process_type so that all are using different CUDA device?

    def generate_model_selection_stage(self):
        td = rp.TaskDescription()
        td.pre_exec = [
                "module load PrgEnv-gnu",
                "export HDF5_USE_FILE_LOCKING=FALSE",
                "module use /soft/modulefiles",
                "module load conda"
                ]
        if self.args.conda_env is not None:
            td.pre_exec.append("conda activate {}".format(self.args.conda_env))
        if self.args.enable_darshan:
            td.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        else:
            td.executable = 'python'
        td.arguments = ['{}/Executables/selection.py'.format(self.args.work_dir),
                       '--phase={}'.format(self.stage_idx),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--write_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["selection"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["selection"]["read"])]
        td.post_exec = []
        td.ranks = 1
        td.cores_per_rank = 8
        td.uid = ru.generate_id(self.TASK_SELECT)
        self._submit_task(td)

    def generate_agent_stage(self):
        td = rp.TaskDescription()
        td.pre_exec = [
                "module load PrgEnv-gnu",
                "export HDF5_USE_FILE_LOCKING=FALSE",
                "module use /soft/modulefiles",
                "module load conda"
                ]
        if self.args.conda_env is not None:
            td.pre_exec.append("conda activate {}".format(self.args.conda_env))
        if self.args.enable_darshan:
            td.executable = 'DARSHAN_EXCLUDE_DIRS=/proc,/etc,/dev,/sys,/snap,/run,/user,/lib,/bin,/lus/grand/projects/CSC249ADCD08/twang/env/rct-recup-polaris/,/grand/CSC249ADCD08/twang/env/rct-recup-polaris/,/tmp LD_PRELOAD=/home/twang3/libraries/darshan/lib/libdarshan.so DARSHAN_ENABLE_NONMPI=1 python'
        else:
            td.executable = 'python'
        td.arguments = ['{}/Executables/agent.py'.format(self.args.work_dir),
                       '--num_epochs={}'.format(self.args.num_epochs_agent),
                       '--device=gpu',
                       '--phase={}'.format(self.stage_idx),
                       '--data_root_dir={}'.format(self.args.data_root_dir),
                       '--model_dir={}'.format(self.args.model_dir),
                       '--num_sample={}'.format(self.args.num_sample),
                       '--num_mult={}'.format(self.args.num_mult_agent),
                       '--num_mult_outlier={}'.format(self.args.num_mult_outlier),
                       '--dense_dim_in={}'.format(self.args.dense_dim_in),
                       '--dense_dim_out={}'.format(self.args.dense_dim_out),
                       '--mat_size={}'.format(self.args.mat_size),
                       '--preprocess_time={}'.format(self.args.preprocess_time_agent),
                       '--write_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["agent"]["write"]),
                       '--read_size={}'.format(self.io_dict["phase{}".format(self.stage_idx)]["agent"]["read"])]
        td.post_exec = []
        td.ranks = 1
        td.cores_per_rank = self.args.num_sim
        td.gpus_per_rank = 1
        td.uid = ru.generate_id(self.TASK_AGENT)
        self._submit_task(td)


if __name__ == "__main__":
    mvp = MVP()
    try:
        mvp.start()
        while True:
            time.sleep(1)
    finally:
        mvp.close()
