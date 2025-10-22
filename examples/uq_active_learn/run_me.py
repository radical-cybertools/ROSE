# run_me.py
import sys
import os
import json
from pathlib import Path
import asyncio
import subprocess
from rose.uq.uq_active_learner import ParallelUQLearner
from rose.metrics import MODEL_ACCURACY, PREDICTIVE_ENTROPY
from rose import TaskConfig
from rose.uq.uq_learner import UQLearnerConfig
from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend
from concurrent.futures import ProcessPoolExecutor
from radical.asyncflow import ConcurrentExecutionBackend

TEST_RADICAL = False
TEST_CUSTOM_UQ = False

if TEST_CUSTOM_UQ:
    UQ_METRIC_NAME = 'custom_uq'   #if you want to use custom metric defined in check_uq.py
else:
    UQ_METRIC_NAME = PREDICTIVE_ENTROPY


ACC_THRESHOLD = 0.5
UQ_THRESHOLD = 1.5
ITERATIONS = 3
PIPELINES = ['UQ1', 'UQ2']
TASK_TYPE = 'classification'
USECASE = 'ENSEMBLE'        
            # Options: 'Bayesian', 'SINGLE_MODEL', 'ENSEMBLE'
UQ_QUERY_SIZE = 1

home_dir = os.environ.get('ROSE_HOME', subprocess.check_output(["pwd"], text=True).strip())

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
    
    if TEST_RADICAL:
        RESOURCES = {
            'runtime': 300, 
            'resource': 'local.localhost', 
            #'resource': 'purdue.anvil',
            'cores': 16
        }
        engine = await RadicalExecutionBackend(RESOURCES)
    else:
        engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())

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
        iteration = kwargs.get("--iteration")
        prediction_dir = kwargs.get("--prediction_dir")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/predict.py --model_name {model_name} ' \
            f'--prediction_dir {prediction_dir} ' \
            f'--iteration {iteration} --learner_name {learner_name} --home_dir {home_dir}'

    # Define and register the active learning task with UQ metrics
    @learner.active_learn_task()
    async def active_learn(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/active_learn.py  --learner_name {learner_name} ' \
                f'--home_dir {home_dir}'

    # Defining the stop criterion with a metric (MODEL_ACCURACY in this case)
    @learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=ACC_THRESHOLD)
    async def check_accuracy(*args, **kwargs):
        model_name = kwargs.get("--model_name")
        home_dir = kwargs.get("--home_dir")
        return f'{code_path}/check_accuracy.py --model_name {model_name} ' \
               f'--home_dir {home_dir}'

    # Defining the stop criterion with a metric (MODEL_ACCURACY in this case)
    @learner.uncertainty_quantification(uq_metric_name=UQ_METRIC_NAME, 
                                        threshold=UQ_THRESHOLD, 
                                        query_size=UQ_QUERY_SIZE)
    async def check_uq(*args, **kwargs):

        home_dir = kwargs.get("--home_dir")
        predict_dir = kwargs.get("--prediction_dir")  
        learner_name = kwargs.get("--learner_name")
        query_size = kwargs.get("--query_size")
        uq_metric_name = kwargs.get("--uq_metric_name")
        task_type = kwargs.get("--task_type")

        return f'{code_path}/check_uq.py --learner_name {learner_name} --predict_dir {predict_dir} ' \
               f'--query_size {query_size} ' \
               f'--uq_metric_name {uq_metric_name} --task_type {task_type} --home_dir {home_dir}'  



    learner_configs = {}
    for PIPELINE in PIPELINES:
        learner_configs[PIPELINE] = UQLearnerConfig(
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
                            '--uq_metric_name': UQ_METRIC_NAME,
                            '--task_type': TASK_TYPE,
                            '--query_size': UQ_QUERY_SIZE,
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