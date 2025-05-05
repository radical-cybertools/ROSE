#!/usr/bin/env python3

import argparse
import os
import sys

from rose.learner import ActiveLearner
from radical.flow import RadicalExecutionBackend, Task

NODES = 4  # TODO: pick this information from the engine directly
CORES_PER_NODE = 56  # TODO: pick this information from the engine directly
RANKS_PER_NODE = 8
RUNTIME = 60  # default value (min)

EXE_BIN = ('/lustre/orion/proj-shared/phy122/rhager/XGC-Devel/'
           '_build2/bin/xgc-eem-cpp-gpu')
RUN_DIR = os.getcwd()


def get_args():
    parser = argparse.ArgumentParser(
        usage='xgc-base.py [<options>]')
    parser.add_argument(
        '-p', '--project',
        dest='project',
        type=str)
    parser.add_argument(
        '-n', '--nodes',
        dest='nodes',
        type=int,
        default=NODES)
    parser.add_argument(
        '-t', '--runtime',
        dest='runtime',
        type=int,
        default=RUNTIME)
    return parser.parse_args(sys.argv[1:])


args = get_args()
engine = RadicalExecutionBackend({'resource': 'ornl.frontier',
                                  'runtime' : args.runtime,
                                  'nodes'   : args.nodes,
                                  'project' : args.project})

acl = ActiveLearner(engine)


# ============================
# Define all utility tasks for the workflow
# ============================

@acl.utility_task
def xgc_base_step(*args):
    return Task(
        executable=EXE_BIN,
        arguments=[],
        pre_launch=[
            f'cd {RUN_DIR}',
            f'sh print_jobinfo.sh ${NODES} {RANKS_PER_NODE} {EXE_BIN}',
            f'sh setup_rundir.sh {RUN_DIR}'
        ],
        pre_exec=[
            f'cd {RUN_DIR}',
            # Load modules used to build XGC as well as runtime environment
            'source modules.sh',
            'module load miniforge3',
            'source activate /lustre/orion/world-shared/phy122/AI-Hackathon_2025/'
            + 'rhager/conda_xgc_pytorch'
        ],
        ranks=NODES * RANKS_PER_NODE,
        # configure ntasks-per-node by using the corresponding number of
        # cores per rank (== `--cpus-per-task`)
        cores_per_rank=CORES_PER_NODE // RANKS_PER_NODE,
        gpus_per_rank=1
    )


# ============================
# Define and run the full workflow
# ============================

def run_workflow():
    """
    Execute the workflow
    """
    step_0 = xgc_base_step()

    print('Results:')
    print([s.result() for s in [step_0]])

    print('Workflow completed successfully!')


if __name__ == '__main__':
    run_workflow()
    engine.shutdown()

