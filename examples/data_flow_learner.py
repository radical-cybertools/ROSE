#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal ROSE simulation-training coupling with ROSEDataClient
"""

import asyncio
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor

from rose import Learner
from rose import DataClient
from radical.asyncflow import WorkflowEngine, ConcurrentExecutionBackend
from radical.asyncflow.logging import init_default_logger

from smartredis import Client


logger = logging.getLogger(__name__)

async def in_memory_learner():

    engine = await ConcurrentExecutionBackend(ProcessPoolExecutor())
    asyncflow = await WorkflowEngine.create(engine)
    learner = Learner(asyncflow)
    init_default_logger(logging.INFO)

    @learner.utility_task(as_executable=False)
    async def simulation(sim_id: int, params: np.ndarray, redis_addr: str):
        """Simulation: takes parameters, produces results"""
        client = Client(address=redis_addr, cluster=False)
        
        # Simulate computation
        result = params * 2.0 + np.random.rand(*params.shape)
        
        # Store simulation output for training
        client.put_tensor(f"sim_output_{sim_id}", result)
        client.put_tensor(f"sim_params_{sim_id}", params)
        
        logger.info(f"Simulation {sim_id} completed")
        return sim_id

    @learner.utility_task(as_executable=False)
    async def training(iteration: int, n_samples: int, redis_addr: str):
        """Training: reads simulation outputs, updates model"""
        client = Client(address=redis_addr, cluster=False)
        training_data = []
        
        # Gather simulation results
        for i in range(n_samples):
            params = client.get_tensor(f"sim_params_{i}")
            output = client.get_tensor(f"sim_output_{i}")
            training_data.append((params, output))
        
        # Train model (simplified)
        model_weights = np.mean([out for _, out in training_data], axis=0)
        
        # Store trained model
        client.put_tensor(f"model_iter_{iteration}", model_weights)
        
        logger.info(f"Training iteration {iteration} completed")
        return model_weights
    
    async def main():
        # Main process client for statistics only
        REDIS_ADDRESS = "127.0.0.1:6379"
        redis = Client(address=REDIS_ADDRESS, cluster=False)
        data_client = DataClient(redis, enable_tracking=True)
        
        n_iterations = 3
        samples_per_iteration = 500
        
        for iteration in range(n_iterations):
            logger.info(f"Starting iteration {iteration}")
            
            # Run simulations in parallel - PASS redis_addr
            params_list = [np.random.rand(100) for _ in range(samples_per_iteration)]
            sim_ids = await asyncio.gather(*[
                simulation(i, params, REDIS_ADDRESS) for i, params in enumerate(params_list)
            ])

            logger.info(sim_ids)
            
            # Train on simulation results - PASS redis_addr
            model = await training(iteration, samples_per_iteration, REDIS_ADDRESS)

            logger.info(model)
        
        # Post-analysis: read back data to populate statistics
        for iteration in range(n_iterations):
            data_client.get_tensor(f"model_iter_{iteration}")
        
        # Analyze coupling pattern
        metrics = await data_client.get_flow_metrics()
        logger.info(f"Total operations: {metrics.total_operations}")
        logger.info(f"Read/Write ratio: {metrics.read_write_ratio:.2f}")
        logger.info(f"Most frequent method: {metrics.most_frequent_method}")
        
        await data_client.export_stats("simulation_training_stats.json")

    try:
        await main()
    finally:
        await learner.shutdown()

if __name__ == "__main__":
    asyncio.run(in_memory_learner())