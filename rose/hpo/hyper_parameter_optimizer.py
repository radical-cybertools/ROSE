from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
import logging
import asyncio

from radical.asyncflow import WorkflowEngine

logger = logging.getLogger(__name__)

class HPOBase(ABC):
    """
    Abstract base class for HPO.

    Users inherit from this class to integrate their HPO tool with ROSE's execution model.
    This class offers lazy config generation to avoid memory bottlenecks with large-scale HPO.
    
    Example of execution flow:

    # Submit ALL N trials at once, but only run 5 at a time
    # As soon as one finishes, report it and start the next

    Trial 0 ████████ (done) → report immediately → suggest Trial 5 → start Trial 5
    Trial 1 ██ (done) → report immediately → suggest Trial 6 → start Trial 6
    Trial 2 ████████████ (still running)
    Trial 3 ████ (done) → report immediately → suggest Trial 7 → start Trial 7
    Trial 4 ██████ (done) → report immediately → suggest Trial 8 → start Trial 8

    """
    
    def __init__(self, 
                 trainable: Callable[[Dict], Dict],
                 resource_requirements: Optional[Dict] = None,
                 config_buffer_size: Optional[int] = None):
        """
        Args:
            trainable: Training function: config -> metrics dict
            resource_requirements: Resources per trial {'gpus': 1, 'ranks': 4}
            config_buffer_size: Max configs to buffer in memory (default: 2 * max_concurrent)
        """
        self.trainable = trainable
        self.resource_requirements = resource_requirements or {'gpus': 1, 'ranks': 1}
        self.config_buffer_size = config_buffer_size
        
        self._iteration = 0
        self._all_results = []
        
    @abstractmethod
    def suggest_configurations(self, n_suggestions: int) -> List[Dict]:
        """
        Suggest next hyperparameter configurations to evaluate.
        
        Args:
            n_suggestions: Number of configurations to suggest
            
        Returns:
            List of config dicts: [{'lr': 0.001, 'batch_size': 32}, ...]
        """
        pass
    
    @abstractmethod
    def report_results(self, results: List[Dict]) -> None:
        """
        Report evaluation results back to HPO algorithm.
        
        Args:
            results: [{'config': {...}, 'score': 0.95, ...}, ...]
        """
        pass
    
    @abstractmethod
    def should_stop(self) -> bool:
        """
        Decide if HPO should terminate.
        
        Returns:
            True if optimization should stop
        """
        pass
    
    def on_trial_start(self, config: Dict, trial_id: int) -> None:
        """Called before trial starts"""
        pass
    
    def on_trial_complete(self, config: Dict, result: Dict, trial_id: int) -> None:
        """Called after trial completes"""
        pass
    
    def get_best_config(self) -> Dict:
        """Get best configuration found"""
        if not self._all_results:
            return {}
        best = max(self._all_results, key=lambda x: x.get('score', float('-inf')))
        return best['config']
    
    def get_best_score(self) -> float:
        """Get best score found"""
        if not self._all_results:
            return float('-inf')
        return max(r.get('score', float('-inf')) for r in self._all_results)
    
    async def run_hpo(self, 
                      asyncflow: WorkflowEngine,
                      max_concurrent_trials: int = 10,
                      total_trials: Optional[int] = None) -> Dict:
        """
        Execute HPO with lazy config generation.
        
        Configs are generated on-demand in small batches to avoid memory
        bottlenecks. Trials execute asynchronously - new trials start as
        soon as resources available.
        
        Args:
            asyncflow: ROSE AsyncFlow WorkflowEngine instance
            max_concurrent_trials: Max parallel trials
            total_trials: Total trial budget (None = unlimited)
            
        Returns:
            {'best_config': {...}, 'best_score': 0.95, 'all_results': [...]}
        """
        
        logger.info(f"Starting HPOLearner")
        logger.info(f"Max concurrent trials: {max_concurrent_trials}")
        logger.info(f"Total trial budget: {total_trials or 'unlimited'}")
        logger.info(f"Resources per trial: {self.resource_requirements}")
        
        trial = Trial(
            asyncflow=asyncflow,
            hpo_strategy=self,
            max_concurrent=max_concurrent_trials
        )

        # Determine buffer size
        buffer_size = self.config_buffer_size or (max_concurrent_trials * 2)
        
        # Small buffer of pending configs (lazy generation)
        pending_trial_configs = []
        
        # Currently running trial tasks
        running_trial_tasks = set()
        
        # Counters
        n_trials_completed = 0
        n_trials_requested = 0
        
        while not self.should_stop():
            if total_trials and n_trials_completed >= total_trials:
                break
            
            # Refill config buffer when running low (lazy generation)
            if len(pending_trial_configs) < buffer_size:
                if not total_trials or n_trials_requested < total_trials:
                    n_needed = buffer_size - len(pending_trial_configs)
                    
                    if total_trials:
                        n_needed = min(n_needed, total_trials - n_trials_requested)
                    
                    if n_needed > 0:
                        new_configs = self.suggest_configurations(n_needed)
                        
                        if not new_configs:
                            if not running_trial_tasks and not pending_trial_configs:
                                break
                        else:
                            pending_trial_configs.extend(new_configs)
                            n_trials_requested += len(new_configs)
                            logger.info(f"Iteration {self._iteration}: Suggested {len(new_configs)} configs")
                            self._iteration += 1
            
            # Submit new trials up to max_concurrent limit
            while pending_trial_configs and len(running_trial_tasks) < max_concurrent_trials:
                config = pending_trial_configs.pop(0)
                trial_id = n_trials_completed + len(running_trial_tasks)
                
                task = asyncio.create_task(
                    trial.execute(
                        config=config,
                        trial_id=trial_id,
                        trainable=self.trainable,
                        resource_requirements=self.resource_requirements,
                        on_trial_start=self.on_trial_start,
                        on_trial_complete=self.on_trial_complete
                    )
                )
                
                running_trial_tasks.add(task)
                logger.info(f"Submitted trial {trial_id} (running: {len(running_trial_tasks)})")
            
            # Wait for at least one trial to complete
            if running_trial_tasks:
                done, running_trial_tasks = await asyncio.wait(
                    running_trial_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process each completed trial
                for task in done:
                    result = await task
                    self._all_results.append(result)
                    n_trials_completed += 1
                    
                    # Report result immediately
                    self.report_results([result])
                    
                    current_best = self.get_best_score()
                    logger.info(f"Trial {result['trial_id']} completed: "
                          f"score={result.get('score', 'N/A'):.4f} | "
                          f"completed={n_trials_completed} | "
                          f"best={current_best:.4f}")
            
            # Stop if nothing running and nothing pending
            if not running_trial_tasks and not pending_trial_configs:
                break
        
        # Wait for remaining running trials
        if running_trial_tasks:
            logger.info(f"Waiting for {len(running_trial_tasks)} remaining trials...")
            remaining_results = await asyncio.gather(*running_trial_tasks)
            
            for result in remaining_results:
                self._all_results.append(result)
                n_trials_completed += 1
                self.report_results([result])
                logger.info(f"Trial {result['trial_id']} completed: "
                      f"score={result.get('score', 'N/A'):.4f}")
        
        logger.info(f"HPO Completed")
        logger.info(f"Total trials: {n_trials_completed}")
        logger.info(f"Best score: {self.get_best_score():.4f}")
        logger.info(f"Best config: {self.get_best_config()}")
        
        return {
            'best_config': self.get_best_config(),
            'best_score': self.get_best_score(),
            'all_results': self._all_results,
            'n_trials': n_trials_completed
        }


class Trial:
    """
    Placeholder for HPO trials
    """
    
    def __init__(self, 
                 asyncflow: WorkflowEngine,
                 hpo_strategy: HPOBase,
                 max_concurrent: int = 10):
        """
        Args:
            asyncflow: AsyncFlow WorkflowEngine instance
            hpo_strategy: User's HPOBase implementation
            max_concurrent: Max parallel trials
        """
        self.asyncflow = asyncflow
        self.hpo_strategy = hpo_strategy
        self.max_concurrent = max_concurrent

    async def execute(self,
                      config: Dict,
                      trial_id: int,
                      trainable: Callable,
                      resource_requirements: Dict,
                      on_trial_start: Callable,
                      on_trial_complete: Callable) -> Dict:
        """
        Execute a single trial.
        """

        if resource_requirements.get('use_executable', False):
            @self.asyncflow.executable_task
            async def execute_trial_task():
                on_trial_start(config, trial_id)
                return trainable(config)
        else:
            @self.asyncflow.function_task
            async def execute_trial_task():
                on_trial_start(config, trial_id)
                
                result = trainable(config)
                
                if not isinstance(result, dict):
                    result = {'score': result}
                
                result['config'] = config
                result['trial_id'] = trial_id
                
                on_trial_complete(config, result, trial_id)
                
                return result

        future = execute_trial_task(task_description=resource_requirements)
        result = await future
        
        return result
