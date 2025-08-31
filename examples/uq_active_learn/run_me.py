# run_me.py
import sys
import os
import numpy as np
import json
from pathlib import Path
import asyncio
import shutil
from rose.uq.uq_learner import ParallelUQLearner
from rose.metrics import MODEL_ACCURACY, PREDICTIVE_ENTROPY
from rose import TaskConfig
from rose import LearnerConfig
from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ConcurrentExecutionBackend
from radical.asyncflow import RadicalExecutionBackend
# from radical.asyncflow import DaskExecutionBackend
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rose.uq import UQMetrics

ACC_THRESHOLD = 0.8
UQ_THRESHOLD = 0
ITERATIONS = 3
PIPELINES = ['UQ1', 'UQ2']
TASK_TYPE = 'classification'
USECASE = 'ENSEMBLE'        
            # Options: 'Bayesian', 'SINGLE_MODEL', 'ENSEMBLE'
UQ_QUERY_SIZE = 10
home_dir = '/anvil/scratch/x-mgoliyad1/uq/ROSE/examples/uq_active_learn'

async def uq_learner():

    if USECASE == 'Bayesian':
        NUM_PREDICTION = 1
        MODELS = ['BayesianNN']
    elif USECASE == 'SINGLE_MODEL':
        NUM_PREDICTION = 2
        MODELS = ['MC_Dropout_CNN']
    elif USECASE == 'ENSEMBLE':
        NUM_PREDICTION = 2
        MODELS = ['MC_Dropout_CNN', 'MC_Dropout_MLP']
    else:
        return
    
    #engine = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    #engine = await RadicalExecutionBackend(RESOURCES, raptor_config)
    #engine = await RadicalExecutionBackend({'resource': 'local.localhost'})
    #engine = await DaskExecutionBackend({'n_workers': 2, 'threads_per_worker': 1})
    asyncflow = await WorkflowEngine.create(engine)

    learner = ParallelUQLearner(asyncflow)

    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @learner.simulation_task()
    async def simulation(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        train_batch = kwargs.get("--train_batch")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/simulation.py --train_batch {train_batch} --learner_name {learner_name} --home_dir {home_dir}'

    # Define and register the training task for each model
    @learner.training_task()
    async def training(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        model_name = kwargs.get("--model_name")
        epochs = kwargs.get("--epochs")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/training.py --model_name {model_name} --learner_name {learner_name} --epochs {epochs} --home_dir {home_dir}'

    # Define and register the predict task for each model
    @learner.prediction_task()
    async def prediction(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        model_name = kwargs.get("--model_name")
        prediction_num = kwargs.get("--prediction_num")
        prediction_dir = kwargs.get("--prediction_dir")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/predict.py --model_name {model_name} --prediction_dir {prediction_dir} --prediction_num {prediction_num}  --learner_name {learner_name} --home_dir {home_dir}'

    # Define and register the active learning task with UQ metrics
    @learner.active_learn_task()
    async def active_learn(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/active_learn.py  --learner_name {learner_name} --home_dir {home_dir}'

    # Defining the stop criterion with a metric (MODEL_ACCURACY in this case)
    @learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=ACC_THRESHOLD)
    async def check_accuracy(*args, **kwargs):
        model_name = kwargs.get("--model_name")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/check_accuracy.py --model_name {model_name} --home_dir {home_dir}'

    # Defining the stop criterion with a metric (MODEL_ACCURACY in this case)
    @learner.uncertainty_quantification(as_executable=False, uq_metric_name=PREDICTIVE_ENTROPY, 
                                        threshold=UQ_THRESHOLD, query_size=UQ_QUERY_SIZE)
    async def check_uq(*args, **kwargs):
        # model_name = kwargs.get("--model_name")
        # return f'{code_path}/check_uq.py --model_name {model_name}'  
        prediction_dir = kwargs.get("--prediction_dir")  
        prediction_dir = Path(prediction_dir)
        learner_name = kwargs.get("--learner_name")

        all_preds = []
        all_files = []
        for file in prediction_dir.iterdir():
            if file.is_file():
                if file.suffix == '.npy':
                    preds = np.load(file, allow_pickle=True)
                    all_preds.append(np.vstack(preds))
                    all_files.append(file)
        all_preds = np.array(all_preds)

        if len(all_preds) == 0:
            return sys.float_info.max
        else:
            uq = UQMetrics(task_type=TASK_TYPE)
            top_idx_local, uq_metric = uq.select_top_uncertain(all_preds, k=UQ_QUERY_SIZE, metric=PREDICTIVE_ENTROPY, plot='plot_scatter_uncertainty')

            with open(Path(home_dir, learner_name + '_uq_selection.json'), 'w') as f:
                json.dump(top_idx_local.tolist(), f)
            return np.mean(uq_metric)

    learner_configs = {}
    for i, PIPELINE in enumerate(PIPELINES):
        learner_configs[PIPELINE] = LearnerConfig(
                            simulation=TaskConfig(kwargs={                            
                            '--home_dir': home_dir,
                            '--train_batch': 50,
                            '--learner_name': f'{PIPELINE}'}),

                            training=TaskConfig(kwargs={                            
                            '--epochs': 10,
                            '--home_dir': home_dir,
                            '--learner_name': f'{PIPELINE}'}),
 
                            prediction=TaskConfig(kwargs={                            
                            '--home_dir': home_dir,
                            '--learner_name': f'{PIPELINE}',
                            '--prediction_dir': f'{PIPELINE}_prediction'}),

                            uncertainty=TaskConfig(kwargs={                            
                            '--uq_metric_name': PREDICTIVE_ENTROPY,
                            '--task_type': TASK_TYPE,
                            '--learner_name': f'{PIPELINE}',
                            '--home_dir': home_dir,
                            '--prediction_dir': f'{PIPELINE}_prediction'}),

                            criterion=TaskConfig(kwargs={
                            '--prediction_dir': f'{PIPELINE}_prediction'}),

                            active_learn=TaskConfig(kwargs={
                            '--learner_name': f'{PIPELINE}',
                            '--home_dir': home_dir,
                            }))

    
    results = await learner.teach(
        learner_names=PIPELINES,
        model_names=MODELS,
        learner_configs=learner_configs,
        max_iter=ITERATIONS, 
        num_predictions=NUM_PREDICTION
    )

    print('Teaching is done with Final Results:')
    print(results)

    with open(Path(os.getcwd(), 'UQ_training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    await learner.shutdown()

if __name__ == "__main__":
    asyncio.run(uq_learner())