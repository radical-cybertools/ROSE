import asyncio
import sys, os

from rose import TaskConfig
from rose import LearnerConfig

from rose.al import ParallelActiveLearner
from rose.metrics import MEAN_SQUARED_ERROR_MSE

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

try:
    import numpy, sklearn
except ImportError:
    print("\nRun 'pip install numpy scikit-learn' to use this example.\n")
    sys.exit(1)

async def run_al_parallel():
    engine = await RadicalExecutionBackend({'resource': 'local.localhost'})
    asyncflow = await WorkflowEngine.create(engine)

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


    # Create adaptive simulation config
    adaptive_sim = al.create_adaptive_schedule('simulation',
        lambda i: {
            'kwargs': {
                '--n_labeled': str(100 + i * 50),  # Increase labeled data each iteration
                '--n_features': 2
            }
        })

    results = await al.teach(
        max_iter=1,
        parallel_learners=2,
        learner_configs=[
            LearnerConfig(simulation=adaptive_sim),
            LearnerConfig(simulation=TaskConfig(kwargs={"--n_labeled": "300",
                                                        "--n_features": 2}))
        ]
    )

    await al.shutdown()

if __name__ == "__main__":
    asyncio.run(run_al_parallel())
