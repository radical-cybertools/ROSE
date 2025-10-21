"""
example_optuna_hpo.py - Optuna with ROSE AsyncFlow
"""

import asyncio
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

import optuna
from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from rose.hpo import HPOBase
from radical.asyncflow.logging import init_default_logger
import logging


logger = logging.getLogger(__name__)

class OptunaAdapter(HPOBase):
    """Adapter for Optuna with ROSE

    Flow of events:
        1 Optuna Study suggests N configs
        N trials execute in parallel on different compute nodes via ROSE
        Each trial can also be multi-node (e.g., 8 nodes per trial)

    Optuna's role:
        Runs in one process (your login node)
        Suggests configs using TPE algorithm
        Updates model when results come back
        Serial decision making

    ROSE's role:
        Takes configs from Optuna
        Distributes trials across HPC nodes
        Manages parallel execution N training models
        Returns results to Optuna

    Without ROSE HPO:
        Optuna trials can not scale to N nodes and N * M GPUs
    """

    def __init__(self, 
                 trainable, 
                 search_space,
                 max_trials=50,
                 direction='maximize',
                 **kwargs):
        super().__init__(trainable, **kwargs)
        
        self.search_space = search_space
        self.max_trials = max_trials
        
        # Create Optuna study
        self.study = optuna.create_study(direction=direction)
        
        # Track trials
        self.pending_trials = []
        self.trial_count = 0
        
    def suggest_configurations(self, n_suggestions):
        """Ask Optuna for next configurations to try"""
        configs = []
        
        for _ in range(n_suggestions):
            if self.trial_count >= self.max_trials:
                break
            
            # Ask Optuna for a trial
            trial = self.study.ask()
            
            # Build config from Optuna trial based on search space
            config = {}
            for param_name, param_spec in self.search_space.items():
                param_type = param_spec['type']
                
                if param_type == 'float':
                    config[param_name] = trial.suggest_float(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'int':
                    config[param_name] = trial.suggest_int(
                        param_name,
                        param_spec['low'],
                        param_spec['high'],
                        log=param_spec.get('log', False)
                    )
                elif param_type == 'categorical':
                    config[param_name] = trial.suggest_categorical(
                        param_name,
                        param_spec['choices']
                    )
                else:
                    raise ValueError(f"Unknown param type: {param_type}")
            
            # Store the Optuna trial with the config
            config['_optuna_trial'] = trial
            configs.append(config)
            
            self.pending_trials.append(trial)
            self.trial_count += 1
        
        return configs
    
    def report_results(self, results):
        """Tell Optuna about completed trial results"""
        for result in results:
            trial = result['config']['_optuna_trial']
            score = result['score']
            
            # Report to Optuna
            self.study.tell(trial, score)
            
            # Remove from pending
            if trial in self.pending_trials:
                self.pending_trials.remove(trial)
    
    def should_stop(self):
        """Stop when max_trials reached"""
        return False
    
    def get_best_config(self):
        """Get best config from Optuna study"""
        if len(self.study.trials) == 0:
            return {}
        
        best_trial = self.study.best_trial
        config = best_trial.params.copy()
        
        # Remove internal Optuna trial object
        config.pop('_optuna_trial', None)
        
        return config
    
    def get_best_score(self):
        """Get best score from Optuna study"""
        if len(self.study.trials) == 0:
            return float('-inf')
        return self.study.best_value


def train_model(config):
    """
    User's ML training function
    
    Simulates training a neural network with different hyperparameters
    """
    lr = config['learning_rate']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    dropout = config['dropout']
    optimizer = config['optimizer']
    
    # Simulate variable training time
    training_time = np.random.uniform(2, 8)
    #time.sleep(training_time)
    
    # Simulate model performance
    score = 0.0
    
    # Learning rate contribution (optimal around 0.001)
    score += 0.3 * np.exp(-((np.log10(lr) + 3) ** 2) / 2)
    
    # Batch size contribution (optimal at 32)
    score += 0.25 * np.exp(-((batch_size - 32) ** 2) / 500)
    
    # Hidden size contribution (optimal at 128)
    score += 0.2 * np.exp(-((hidden_size - 128) ** 2) / 5000)
    
    # Dropout contribution (optimal around 0.3)
    score += 0.15 * np.exp(-((dropout - 0.3) ** 2) / 0.1)
    
    # Optimizer contribution
    if optimizer == 'adam':
        score += 0.1
    elif optimizer == 'sgd':
        score += 0.05
    else:
        score += 0.02
    
    # Add noise
    score += np.random.normal(0, 0.03)
    
    # Clip to [0, 1]
    score = np.clip(score, 0, 1)
    
    return {
        'score': score,
        'training_time': training_time,
        'loss': 1 - score
    }


async def main():
    # Create AsyncFlow backend and workflow engine
    init_default_logger(logging.INFO)
    backend = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=20))
    asyncflow = await WorkflowEngine.create(backend=backend)
    
    logger.info("ROSE HPO Example - Optuna Integration\n")
    
    # Define search space for Optuna
    search_space = {
        'learning_rate': {
            'type': 'float',
            'low': 1e-5,
            'high': 1e-1,
            'log': True  # Log scale
        },
        'batch_size': {
            'type': 'categorical',
            'choices': [16, 32, 64, 128]
        },
        'hidden_size': {
            'type': 'int',
            'low': 64,
            'high': 512
        },
        'dropout': {
            'type': 'float',
            'low': 0.0,
            'high': 0.5
        },
        'optimizer': {
            'type': 'categorical',
            'choices': ['adam', 'sgd', 'rmsprop']
        }
    }
    
    # Create Optuna adapter
    hpo = OptunaAdapter(
        trainable=train_model,
        search_space=search_space,
        max_trials=50,
        direction='maximize',  # Maximize score
        resource_requirements={'gpus': 1, 'ranks': 1},
        config_buffer_size=20
    )
    
    # Run HPO on HPC
    results = await hpo.run_hpo(
        asyncflow=asyncflow,
        max_concurrent_trials=10,
        total_trials=50
    )
    
    # logger.info results
    logger.info("BEST HYPERPARAMETERS FOUND (via Optuna)")
    best_config = results['best_config']
    for param, value in best_config.items():
        if isinstance(value, float):
            logger.info(f"  {param}: {value:.6f}")
        else:
            logger.info(f"  {param}: {value}")
    
    logger.info(f"Best Score: {results['best_score']:.4f}")
    logger.info(f"Total Trials: {results['n_trials']}")
    
    # Optuna-specific info
    logger.info("OPTUNA STUDY STATISTICS")
    logger.info(f"Number of finished trials: {len(hpo.study.trials)}")
    logger.info(f"Number of pruned trials: {len([t for t in hpo.study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logger.info(f"Number of complete trials: {len([t for t in hpo.study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Top 5 trials
    logger.info("Top 5 Trials:")
    sorted_trials = sorted(hpo.study.trials, key=lambda t: t.value if t.value else float('-inf'), reverse=True)
    for i, trial in enumerate(sorted_trials[:5]):
        logger.info(f"{i+1}. Trial {trial.number}: score={trial.value:.4f}")
        logger.info(f"lr={trial.params['learning_rate']:.6f}, "
              f"batch={trial.params['batch_size']}, "
              f"hidden={trial.params['hidden_size']}")
    
    # Optuna has built-in visualization (optional)
    import optuna.visualization as vis
    fig = vis.plot_optimization_history(hpo.study)
    fig.write_image("optimization_history.png")
    
    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
