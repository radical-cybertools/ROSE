"""
ROSE HPO Base Module - General purpose HPO infrastructure for HPC using AsyncFlow
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

from radical.asyncflow import (
    WorkflowEngine,
)

import asyncio


class HPOBase(ABC):
    """
    Abstract base class for HPO on HPC via ROSE AsyncFlow.
    
    Users inherit from this class to integrate their favorite HPO tool
    (Ray Tune, Optuna, custom algorithms) with ROSE's HPC execution.
    
    Contract:
    - User implements: HPO algorithm logic (what configs to try, when to stop)
    - ROSE provides: HPC execution via AsyncFlow
    """
    
    def __init__(self, 
                 trainable: Callable[[Dict], Dict],
                 resource_requirements: Optional[Dict] = None):
        """
        Args:
            trainable: Training function: config -> metrics dict
            resource_requirements: Resources per trial:
                {'gpus': 1, 'ranks': 4, 'cores_per_rank': 8}
        """
        self.trainable = trainable
        self.resource_requirements = resource_requirements or {'gpus': 1, 'ranks': 1}
        
        # Internal state
        self._iteration = 0
        self._all_results = []
        self._hpo_learner = None
        
    # =========================================================================
    # Abstract methods - USER MUST IMPLEMENT
    # =========================================================================
    
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
        Report evaluation results back to your HPO algorithm.
        
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
    
    # =========================================================================
    # Optional callbacks
    # =========================================================================
    
    def on_trial_start(self, config: Dict, trial_id: int) -> None:
        """Called before trial starts (optional)"""
        pass
    
    def on_trial_complete(self, config: Dict, result: Dict, trial_id: int) -> None:
        """Called after trial completes (optional)"""
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
        Execute HPO on HPC using ROSE AsyncFlow.
        
        Args:
            asyncflow: ROSE AsyncFlow WorkflowEngine instance
            max_concurrent_trials: Max parallel trials
            total_trials: Total trial budget (None = unlimited)
            
        Returns:
            {'best_config': {...}, 'best_score': 0.95, 'all_results': [...]}
        """
        
        print(f"Starting HPO on HPC via ROSE AsyncFlow")
        print(f"   Max concurrent trials: {max_concurrent_trials}")
        print(f"   Total trial budget: {total_trials or 'unlimited'}")
        print(f"   Resources per trial: {self.resource_requirements}")
        
        # Create HPO learner
        self._hpo_learner = HPOLearner(
            asyncflow=asyncflow,
            hpo_strategy=self,
            max_concurrent=max_concurrent_trials
        )
        
        # Optimization loop
        n_trials_completed = 0
        
        while not self.should_stop():
            if total_trials and n_trials_completed >= total_trials:
                break
            
            # How many configs to suggest this iteration
            n_suggest = min(
                max_concurrent_trials,
                (total_trials - n_trials_completed) if total_trials else max_concurrent_trials
            )
            
            # User's HPO algorithm suggests configs
            configs = self.suggest_configurations(n_suggest)
            
            if not configs:
                print("⚠️  No more configurations suggested, stopping")
                break
            
            print(f"\n📊 Iteration {self._iteration}: Evaluating {len(configs)} configs")
            
            # Execute trials on HPC via AsyncFlow
            batch_results = await self._hpo_learner.execute_trials(
                configs=configs,
                trainable=self.trainable,
                resource_requirements=self.resource_requirements,
                on_trial_start=self.on_trial_start,
                on_trial_complete=self.on_trial_complete,
                iteration=self._iteration
            )
            
            # Store results
            self._all_results.extend(batch_results)
            n_trials_completed += len(batch_results)
            
            # User's HPO algorithm processes results
            self.report_results(batch_results)
            
            # Progress
            current_best = self.get_best_score()
            print(f"   Completed: {n_trials_completed} trials")
            print(f"   Best score so far: {current_best:.4f}")
            
            self._iteration += 1
        
        print(f"\n✨ HPO Complete!")
        print(f"   Total trials: {n_trials_completed}")
        print(f"   Best score: {self.get_best_score():.4f}")
        print(f"   Best config: {self.get_best_config()}")
        
        return {
            'best_config': self.get_best_config(),
            'best_score': self.get_best_score(),
            'all_results': self._all_results,
            'n_trials': n_trials_completed
        }


class HPOLearner:
    """
    Executes HPO trials on HPC using ROSE AsyncFlow.
    
    This is internal infrastructure - users don't interact with this directly.
    """
    
    def __init__(self, asyncflow, hpo_strategy: HPOBase, max_concurrent: int = 10):
        """
        Args:
            asyncflow: AsyncFlow WorkflowEngine instance
            hpo_strategy: User's HPOBase implementation
            max_concurrent: Max parallel trials
        """
        self.asyncflow: WorkflowEngine = asyncflow
        self.hpo_strategy = hpo_strategy
        self.max_concurrent = max_concurrent
        
    async def execute_trials(self,
                            configs: List[Dict],
                            trainable: Callable,
                            resource_requirements: Dict,
                            on_trial_start: Callable,
                            on_trial_complete: Callable,
                            iteration: int) -> List[Dict]:
        """
        Execute batch of trials on HPC using AsyncFlow.
        
        All trials run in parallel across HPC resources.
        """
        
        # Define AsyncFlow task for single trial
        if resource_requirements.get('use_executable', False):
            # For executable-based training
            @self.asyncflow.executable_task
            async def execute_trial(config: Dict, trial_id: int):
                """Single trial as executable"""
                on_trial_start(config, trial_id)
                # Return executable command
                return trainable(config)  # Should return executable path
        else:
            # For Python function-based training
            @self.asyncflow.function_task
            async def execute_trial(config: Dict, trial_id: int):
                """Single trial as Python function"""
                on_trial_start(config, trial_id)
                
                print(f"    🔧 Trial {trial_id}: {config}")
                
                # Execute user's training function
                result = trainable(config)
                
                # Ensure result format
                if not isinstance(result, dict):
                    result = {'score': result}
                
                result['config'] = config
                result['trial_id'] = trial_id
                
                on_trial_complete(config, result, trial_id)
                
                print(f"    ✓ Trial {trial_id}: score={result.get('score', 'N/A')}")
                
                return result
        
        # Submit all trials to AsyncFlow with resource requirements
        trial_futures = []
        
        for i, config in enumerate(configs):
            trial_id = len(self.hpo_strategy._all_results) + i
            
            # Submit trial with HPC resource requirements
            future = execute_trial(
                config, 
                trial_id,
                task_description=resource_requirements  # AsyncFlow keyword
            )
            
            trial_futures.append(future)
        
        # Wait for all trials to complete (AsyncFlow handles parallelism)
        results = await asyncio.gather(*trial_futures)
        
        return results
