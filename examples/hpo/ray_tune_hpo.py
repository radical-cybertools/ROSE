"""
example_ray_tune_hpo.py - Ray Tune with ROSE AsyncFlow
"""

import asyncio
import numpy as np
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger
from concurrent.futures import ThreadPoolExecutor
import logging
from rose.hpo import HPOBase

logger = logging.getLogger(__name__)
# ============================================================================
# User's Ray Tune Adapter
# ============================================================================

class RayTuneAdapter(HPOBase):
    """Adapter for Ray Tune with ROSE"""
    
    def __init__(self, trainable, search_space, max_trials=50, **kwargs):
        super().__init__(trainable, **kwargs)
        
        self.search_space = search_space
        self.max_trials = max_trials
        self.trial_count = 0
        
        # Ray Tune scheduler
        self.scheduler = ASHAScheduler(
            metric='score',
            mode='max',
            max_t=100,
            grace_period=10
        )
        
    def suggest_configurations(self, n_suggestions):
        """Sample from Ray Tune search space"""
        configs = []
        
        for _ in range(n_suggestions):
            if self.trial_count >= self.max_trials:
                break
            
            config = {}
            for param, spec in self.search_space.items():
                if hasattr(spec, 'sample'):
                    config[param] = spec.sample()
                elif isinstance(spec, list):
                    config[param] = np.random.choice(spec)
                else:
                    config[param] = spec
            
            configs.append(config)
            self.trial_count += 1
        
        return configs
    
    def report_results(self, results):
        """Report to Ray Tune scheduler"""
        pass
    
    def should_stop(self):
        return self.trial_count >= self.max_trials


# ============================================================================
# Training Function
# ============================================================================

def train_pytorch_model(config):
    """Simulated PyTorch training"""
    lr = config['learning_rate']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    dropout = config['dropout']
    
    score = (
        0.3 * np.exp(-((np.log10(lr) + 3) ** 2) / 2) +
        0.3 * np.exp(-((batch_size - 32) ** 2) / 500) +
        0.2 * np.exp(-((hidden_size - 128) ** 2) / 5000) +
        0.2 * np.exp(-((dropout - 0.3) ** 2) / 0.1) +
        np.random.normal(0, 0.03)
    )
    
    return {'score': score, 'epochs': 50}


# ============================================================================
# Main
# ============================================================================

async def main():
    # Create AsyncFlow
    init_default_logger(logging.INFO)
    backend = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=20))
    flow = await WorkflowEngine.create(backend=backend)
    
    print("ROSE HPO Example - Ray Tune Integration\n")
    
    # Create Ray Tune adapter
    hpo = RayTuneAdapter(
        trainable=train_pytorch_model,
        search_space={
            'learning_rate': tune.loguniform(1e-5, 1e-1),
            'batch_size': tune.choice([16, 32, 64, 128]),
            'hidden_size': tune.choice([64, 128, 256, 512]),
            'dropout': tune.uniform(0.0, 0.5)
        },
        max_trials=50,
        resource_requirements={'gpus': 1, 'ranks': 1}
    )
    
    # Run HPO (async only)
    results = await hpo.run_hpo(
        asyncflow=flow,
        max_concurrent_trials=10,
        total_trials=50
    )
    
    # Results
    logger.info("BEST HYPERPARAMETERS FOUND")
    for param, value in results['best_config'].items():
        logger.info(f"{param}: {value}")
    logger.info(f"Best Score: {results['best_score']:.4f}")
    
    await flow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
