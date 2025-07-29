# run_me.py
import sys
import os
import json
from pathlib import Path
import asyncio
from rose.uq.uq_learner import ParallelUQLearner
from rose.metrics import MODEL_ACCURACY
from rose import TaskConfig
from rose import LearnerConfig
from radical.asyncflow import WorkflowEngine
from radical.asyncflow import RadicalExecutionBackend

ACC_THRESHOLD = 0.8
NUM_ITER = 3
PIPELINES = ['UQ1'] #, 'UQ2']
TASK_TYPE = 'classification'

USECASE = 'SINGLE_MODEL'        
                            # Options: 'Bayesian', 'SINGLE_MODEL', 'ENSEMBLE'
UQ_METRIC = 'entropy'       
                            # Options for TASK_TYPE == classification: 'entropy', 'mutual_information', 'variation_ratio', 'margin'
                            # Options for TASK_TYPE == regression:     'variance', 'interval_width'


async def uq_learner():

    if USECASE == 'Bayesian':
        NUM_PREDICTION = 1
        MODELS = ['BayesianNN']
    elif USECASE == 'SINGLE_MODEL':
        NUM_PREDICTION = 2
        MODELS = ['MC_Dropout_CNN']
    elif USECASE == 'ENSEMBLE':
        NUM_PREDICTION = 1
        MODELS = ['MC_Dropout_CNN', 'MC_Dropout_MLP']
    else:
        return
    
    engine = RadicalExecutionBackend({'resource': 'local.localhost'})
    asyncflow = WorkflowEngine(engine)

    learner = ParallelUQLearner(asyncflow)

    code_path = f'{sys.executable} {os.getcwd()}'

    # Define and register the simulation task
    @learner.simulation_task
    async def simulation(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        n_labeled = kwargs.get("--n_labeled")
        return f'{code_path}/simulation.py --n_labeled {n_labeled} --learner_name {learner_name}'

    for model_name in MODELS:
        # Define and register the training task for each model
        @learner.training_task(name=model_name)
        async def training(*args, **kwargs):
            learner_name = kwargs.get("--learner_name")
            model_name = kwargs.get("--model_name")
            epochs = kwargs.get("--epochs")
            return f'{code_path}/training.py --model_name {model_name} --learner_name {learner_name} --epochs {epochs}'

        # Define and register the predict task for each model
        @learner.predict_task(name=model_name)
        async def predict(*args, **kwargs):
            learner_name = kwargs.get("--learner_name")
            model_name = kwargs.get("--model_name")
            prediction_dir = kwargs.get("--prediction_dir")
            iteration = kwargs.get("--iteration", 1)
            return f'{code_path}/predict.py --model_name {model_name} --prediction_dir {prediction_dir} --iteration {iteration}  --learner_name {learner_name}'

    # Define and register the active learning task with UQ metrics
    @learner.active_learn_task
    async def active_learn(*args, **kwargs):
        learner_name = kwargs.get("--learner_name")
        uq_metric = kwargs.get("--uq_metric")
        task_type = kwargs.get("--task_type")
        prediction_dir = kwargs.get("--prediction_dir")
        return f'{code_path}/active_learn.py --uq_metric {uq_metric} --task_type {task_type} --prediction_dir {prediction_dir}  --learner_name {learner_name}'

    # Defining the stop criterion with a metric (MODEL_ACCURACY in this case)
    @learner.as_stop_criterion(metric_name=MODEL_ACCURACY, threshold=ACC_THRESHOLD)
    async def check_accuracy(*args, **kwargs):
        model_name = kwargs.get("--model_name")
        return f'{code_path}/check_accuracy.py --model_name {model_name}'
    
    
    learner_configs = {}
    for i, PIPELINE in enumerate(PIPELINES):
        learner_configs[PIPELINE] = LearnerConfig(
                            simulation=TaskConfig(kwargs={                            
                            '--n_labeled': (i+1) * 100,
                            '--learner_name': f'{PIPELINE}'}),

                            training=TaskConfig(kwargs={                            
                            '--epochs': 10,
                            '--learner_name': f'{PIPELINE}'}),

                            active_learn=TaskConfig(kwargs={
                            '--uq_metric': UQ_METRIC,
                            '--task_type': TASK_TYPE,
                            '--learner_name': f'{PIPELINE}',
                            '--prediction_dir': f'{PIPELINE}_prediction'}))
    
    results = await learner.teach(
        learner_names=PIPELINES,
        learner_configs=learner_configs,
        max_iter=NUM_ITER, 
        num_predictions=NUM_PREDICTION
    )


    print(results)

    with open(Path(os.getcwd(), 'UQ_training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    await learner.shutdown()

if __name__ == "__main__":
    asyncio.run(uq_learner())
    
