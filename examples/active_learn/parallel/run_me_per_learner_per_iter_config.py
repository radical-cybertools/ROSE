import asyncio
import sys, os

from rose import TaskConfig
from rose import LearnerConfig

from rose.al import ParallelActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend


async def run_al_parallel():
    engine = RadicalExecutionBackend({'resource': 'local.localhost'})
    asyncflow = WorkflowEngine(engine)
    
    al = ParallelActiveLearner(asyncflow)
    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @al.simulation_task
    async def simulation(*args, **kwargs):
        n_labeled = kwargs.get("--n_labeled", 100)
        n_features = kwargs.get("--n_features", 2)

        return f"{code_path}/sim.py --n_labeled {n_labeled} --n_features {n_features}"

    # Define and register the training task
    @al.training_task
    async def training(*args, **kwargs):
        return f'{code_path}/train.py'

    # Define and register the active learning task
    @al.active_learn_task
    async def active_learn(*args, **kwargs):
        return f'{code_path}/active.py'

    # Defining the stop criterion with a metric (MSE in this case)
    @al.as_stop_criterion(metric_name=MEAN_SQUARED_ERROR_MSE, threshold=0.1)
    async def check_mse(*args, **kwargs):
        return f'{code_path}/check_mse.py'

    # Run with custom configs
    results = await al.teach(
        parallel_learners=3,
        learner_configs=[
            # Learner 0: Same config for all iterations (your current pattern)
            LearnerConfig(simulation=TaskConfig(kwargs={"--n_labeled": "200",
                                                                "--n_features": 2})),

            # Learner 1: Different configs per iteration
            LearnerConfig(
                simulation={
                    0: TaskConfig(kwargs={"--n_labeled": "100", "--n_features": 2}),
                    5: TaskConfig(kwargs={"--n_labeled": "200", "--n_features": 2}),
                    10: TaskConfig(kwargs={"--n_labeled": "400", "--n_features": 2}),
                    -1: TaskConfig(kwargs={"--n_labeled": "500", "--n_features": 2})  # default
                }
            ),
            None,
        ]
    )

    await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(run_al_parallel())