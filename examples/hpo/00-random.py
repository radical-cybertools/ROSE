"""
example_simple_hpo.py - Simple random search with lazy config generation
"""

import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from rose.hpo import HPOBase

from radical.asyncflow.logging import init_default_logger
import logging

logger = logging.getLogger(__name__)

class SimpleRandomSearch(HPOBase):
    """Simple random search HPO"""
    
    def __init__(self, trainable, search_space, n_trials=20, **kwargs):
        super().__init__(trainable, **kwargs)
        self.search_space = search_space
        self.n_trials = n_trials
        self.trials_done = 0
        
    def suggest_configurations(self, n_suggestions):
        configs = []
        
        for _ in range(n_suggestions):
            if self.trials_done >= self.n_trials:
                break
            
            config = {}
            for param, (low, high) in self.search_space.items():
                if isinstance(low, int) and isinstance(high, int):
                    config[param] = np.random.randint(low, high)
                else:
                    config[param] = np.random.uniform(low, high)
            
            configs.append(config)
            self.trials_done += 1
        
        return configs
    
    def report_results(self, results):
        pass
    
    def should_stop(self):
        return self.trials_done >= self.n_trials


def train_model(config):
    """User's ML training function"""
    lr = config['learning_rate']
    batch_size = config['batch_size']
    hidden_size = config['hidden_size']
    
    import time
    time.sleep(np.random.uniform(1, 3))

    score = 0.0
    score += 0.4 * np.exp(-((np.log10(lr) + 3) ** 2) / 2)
    score += 0.3 * np.exp(-((batch_size - 32) ** 2) / 500)
    score += 0.3 * np.exp(-((hidden_size - 128) ** 2) / 5000)
    score += np.random.normal(0, 0.02)
    
    return {'score': score, 'loss': 1 - score}


async def main():
    init_default_logger(logging.INFO)
    backend = await ConcurrentExecutionBackend(ThreadPoolExecutor(max_workers=10))
    asyncflow = await WorkflowEngine.create(backend=backend)
    
    logger.info("ROSE HPO Example - Random Search with Lazy Config Generation\n")
    
    hpo = SimpleRandomSearch(
        trainable=train_model,
        search_space={
            'learning_rate': (1e-5, 1e-1),
            'batch_size': (16, 128),
            'hidden_size': (64, 512)
        },
        n_trials=20,
        resource_requirements={'gpus': 1, 'ranks': 1},
        config_buffer_size=10  # Only buffer 10 configs at a time
    )
    
    results = await hpo.optimize(
        asyncflow=asyncflow,
        max_concurrent_trials=5,
        total_trials=20
    )

    logger.info("FINAL RESULTS")
    logger.info(f"Best Configuration: {results['best_config']}")
    logger.info(f"Best Score: {results['best_score']:.4f}")
    logger.info(f"Total Trials: {results['n_trials']}")

    await asyncflow.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
