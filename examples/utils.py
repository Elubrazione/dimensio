import numpy as np
from typing import Callable, Optional
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformIntegerHyperparameter,
    UniformFloatHyperparameter,
    CategoricalHyperparameter
)
from openbox.utils.history import History, Observation
from openbox.utils.constants import SUCCESS


def create_simple_config_space(n_float: int = 5, n_int: int = 5, seed: int = 42) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    
    for i in range(n_float):
        cs.add_hyperparameter(
            UniformFloatHyperparameter(f'float_param_{i}', 0.0, 10.0)
        )
    
    for i in range(n_int):
        cs.add_hyperparameter(
            UniformIntegerHyperparameter(f'int_param_{i}', 1, 100)
        )
    
    return cs


def create_spark_config_space(seed: int = 42) -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=seed)
    
    # Executor configuration
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.executor.cores', lower=1, upper=8, default_value=4))
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.executor.memory', lower=1024, upper=16384, default_value=4096))
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.executor.instances', lower=1, upper=20, default_value=4))
    
    # Shuffle configuration
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.sql.shuffle.partitions', lower=50, upper=1000, default_value=200))
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'spark.shuffle.compress', lower=0.0, upper=1.0, default_value=1.0))
    
    # Memory management
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'spark.memory.fraction', lower=0.3, upper=0.9, default_value=0.6))
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'spark.memory.storageFraction', lower=0.1, upper=0.7, default_value=0.5))
    
    # Parallelism
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.default.parallelism', lower=50, upper=500, default_value=200))
    
    # Network
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.network.timeout', lower=60, upper=600, default_value=120))
    
    # Broadcast
    cs.add_hyperparameter(UniformIntegerHyperparameter(
        'spark.sql.autoBroadcastJoinThreshold', lower=1, upper=100, default_value=10))
    
    # Compression
    cs.add_hyperparameter(CategoricalHyperparameter(
        'spark.io.compression.codec', choices=['lz4', 'snappy', 'zstd'], default_value='lz4'))
    
    # Adaptive execution
    cs.add_hyperparameter(UniformFloatHyperparameter(
        'spark.sql.adaptive.enabled', lower=0.0, upper=1.0, default_value=1.0))
    
    return cs


def simple_objective(config_dict: dict) -> float:
    score = 100.0
    
    if 'float_param_0' in config_dict:
        score += abs(config_dict['float_param_0'] - 5.0) * 10
    
    if 'int_param_0' in config_dict:
        score += abs(config_dict['int_param_0'] - 50) * 0.5
    
    score += np.random.normal(0, 5)
    
    return max(score, 10.0)


def spark_objective(config_dict: dict) -> float:
    score = 100.0
    
    if 'spark.executor.cores' in config_dict and 'spark.executor.memory' in config_dict:
        cores = config_dict['spark.executor.cores']
        memory = config_dict['spark.executor.memory']
        ratio = memory / (cores * 1024)
        ideal_ratio = 4.0
        ratio_penalty = abs(ratio - ideal_ratio) * 10
        score += ratio_penalty
    
    if 'spark.sql.shuffle.partitions' in config_dict:
        partitions = config_dict['spark.sql.shuffle.partitions']
        if partitions < 200:
            score += (200 - partitions) * 0.5
        elif partitions > 500:
            score += (partitions - 500) * 0.3
    
    if 'spark.memory.fraction' in config_dict:
        fraction = config_dict['spark.memory.fraction']
        score += abs(fraction - 0.65) * 50
    
    score += np.random.normal(0, 10)
    
    return max(score, 10.0)


def generate_history(
    config_space: ConfigurationSpace,
    n_samples: int = 50,
    task_id: str = 'task',
    objective_func: Optional[Callable] = None,
    verbose: bool = True
) -> History:
    return generate_mock_history(config_space, n_samples, task_id, objective_func, verbose)


def generate_mock_history(
    config_space: ConfigurationSpace,
    n_samples: int = 50,
    task_id: str = 'task',
    objective_func: Optional[Callable] = None,
    verbose: bool = True
) -> History:
    if verbose:
        print(f"\n📊 Generating mock history data: {n_samples} samples")
    
    history = History(
        task_id=task_id,
        num_objectives=1,
        num_constraints=0,
        config_space=config_space
    )
    
    if objective_func is None:
        objective_func = simple_objective
    
    observations = []
    for _ in range(n_samples):
        config = config_space.sample_configuration()
        obj_value = objective_func(config.get_dictionary())
        
        obs = Observation(
            config=config,
            objectives=[obj_value],
            constraints=None,
            trial_state=SUCCESS,
            elapsed_time=0.1
        )
        observations.append(obs)
    
    history.update_observations(observations)
    
    if verbose:
        objectives = [obs.objectives[0] for obs in observations]
        print(f"  - Objective range: [{min(objectives):.2f}, {max(objectives):.2f}]")
        print(f"  - Best value: {min(objectives):.2f}")
        print(f"  - Mean value: {np.mean(objectives):.2f}")
    
    return history


def generate_improving_history(
    config_space: ConfigurationSpace,
    n_samples: int = 5,
    iteration: int = 0,
    task_id: str = 'task',
    objective_func: Optional[Callable] = None,
    improvement_rate: float = 3.0,
    verbose: bool = False
) -> History:
    if verbose:
        print(f"📊 Generating improving history (iteration {iteration})")
    
    history = History(
        task_id=task_id,
        num_objectives=1,
        num_constraints=0,
        config_space=config_space
    )
    
    if objective_func is None:
        objective_func = simple_objective
    
    observations = []
    for i in range(n_samples):
        config = config_space.sample_configuration()
        obj_value = objective_func(config.get_dictionary())
        
        if i == 0:
            obj_value -= (iteration + 1) * improvement_rate
            obj_value -= np.random.uniform(1, 3)
        else:
            obj_value -= iteration * improvement_rate
            obj_value += np.random.normal(0, 1)
        
        obj_value = max(obj_value, 10.0)
        
        obs = Observation(
            config=config,
            objectives=[obj_value],
            constraints=None,
            trial_state=SUCCESS,
            elapsed_time=0.1
        )
        observations.append(obs)
    
    history.update_observations(observations)
    
    return history

