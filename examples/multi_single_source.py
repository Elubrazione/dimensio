#!/usr/bin/env python3
"""
Transfer Learning with Multiple Source Tasks Example

This example demonstrates how to leverage historical data from multiple source tasks
to accelerate optimization on a new target task using transfer learning.

Scenario: Optimizing Spark configuration for different workload types
- Source Task 1: Sort workload (past optimization data)
- Source Task 2: Join workload (past optimization data)  
- Source Task 3: Aggregate workload (past optimization data)
- Target Task: Group-by workload (new task we want to optimize)
"""

import numpy as np
from ConfigSpace import ConfigurationSpace
from openbox.utils.history import History

from dimensio import (
    Compressor,
    SHAPBoundaryRangeStep,
    SHAPDimensionStep,
    BoundaryRangeStep,
    setup_logging
)
from dimensio.viz import visualize_compression_details
from utils import create_spark_config_space, generate_mock_history

res_dir = "./results/multiple_single_source"


def objective_function(config: dict, workload_type: str) -> float:
    """
    Simulate objective function for different workload types.
    Lower is better (execution time in seconds).
    
    Different workloads have different parameter sensitivities:
    - Sort: sensitive to shuffle.partitions, parallelism
    - Join: sensitive to executor.memory, autoBroadcastJoinThreshold
    - Aggregate: sensitive to memory.fraction, shuffle.compress
    - Group-by: mixed sensitivity (target task)
    """
    score = 100.0
    
    # Common factors
    cores = config.get('spark.executor.cores', 4)
    memory = config.get('spark.executor.memory', 4096)
    
    resource_score = abs(cores * memory - 16384) * 0.01
    score += resource_score
    
    if workload_type == 'sort':
        # Sort is very sensitive to shuffle and parallelism
        partitions = config.get('spark.sql.shuffle.partitions', 200)
        parallelism = config.get('spark.default.parallelism', 200)
        score += abs(partitions - 400) * 0.5  # Optimal: 400
        score += abs(parallelism - 300) * 0.3  # Optimal: 300
        
    elif workload_type == 'join':
        # Join is sensitive to memory and broadcast threshold
        broadcast = config.get('spark.sql.autoBroadcastJoinThreshold', 10)
        mem_fraction = config.get('spark.memory.fraction', 0.6)
        score += abs(broadcast - 50) * 2.0  # Optimal: 50MB
        score += abs(mem_fraction - 0.7) * 100  # Optimal: 0.7
        
    elif workload_type == 'aggregate':
        # Aggregate is sensitive to memory management and compression
        mem_fraction = config.get('spark.memory.fraction', 0.6)
        storage_fraction = config.get('spark.memory.storageFraction', 0.5)
        shuffle_compress = config.get('spark.shuffle.compress', 1.0)
        score += abs(mem_fraction - 0.65) * 80  # Optimal: 0.65
        score += abs(storage_fraction - 0.3) * 50  # Optimal: 0.3
        score += (1.0 - shuffle_compress) * 30  # Compression helps
        
    elif workload_type == 'groupby':
        # Group-by combines aspects of all workloads (target task)
        partitions = config.get('spark.sql.shuffle.partitions', 200)
        mem_fraction = config.get('spark.memory.fraction', 0.6)
        parallelism = config.get('spark.default.parallelism', 200)
        broadcast = config.get('spark.sql.autoBroadcastJoinThreshold', 10)
        
        score += abs(partitions - 350) * 0.4  # Optimal: 350
        score += abs(mem_fraction - 0.68) * 70  # Optimal: 0.68
        score += abs(parallelism - 250) * 0.25  # Optimal: 250
        score += abs(broadcast - 30) * 1.5  # Optimal: 30MB
    
    score += np.random.normal(0, 5)    
    return max(score, 10.0)


def generate_workload_history(config_space: ConfigurationSpace, 
                              workload_type: str, 
                              n_samples: int = 50) -> History:
    """Generate historical optimization data for a specific workload."""
    print(f"  Generating {n_samples} samples for {workload_type} workload...")
    
    def workload_objective(config_dict):
        return objective_function(config_dict, workload_type)
    
    history = generate_mock_history(
        config_space=config_space,
        n_samples=n_samples,
        task_id=f'spark_{workload_type}',
        objective_func=workload_objective,
        verbose=False  
    )
    
    best_obj = min([obs.objectives[0] for obs in history.observations])
    mean_obj = np.mean([obs.objectives[0] for obs in history.observations])
    print(f"    Best: {best_obj:.2f}, Mean: {mean_obj:.2f}")
    
    return history


def calculate_workload_similarities(target_workload: str) -> dict:
    """
    Calculate similarity scores between source workloads and target workload.
    
    In practice, these could be computed based on:
    - Workload characteristics (data size, operation types, etc.)
    - Historical performance patterns
    - Meta-features
    
    Here we use domain knowledge for demonstration.
    """
    # Similarity matrix (based on parameter sensitivity patterns)
    similarity_matrix = {
        'groupby': {
            'sort': 0.65,      # Moderate similarity (both use shuffle heavily)
            'join': 0.80,      # High similarity (both memory-intensive)
            'aggregate': 0.75  # High similarity (both compute-intensive)
        },
        'sort': {
            'sort': 1.0,
            'join': 0.45,
            'aggregate': 0.50
        },
        'join': {
            'sort': 0.45,
            'join': 1.0,
            'aggregate': 0.60
        },
        'aggregate': {
            'sort': 0.50,
            'join': 0.60,
            'aggregate': 1.0
        }
    }
    
    source_workloads = ['sort', 'join', 'aggregate']
    similarities = {}
    
    for idx, source in enumerate(source_workloads):
        similarities[idx] = similarity_matrix[target_workload][source]
    
    return similarities


def main():
    """Run transfer learning example with multiple source tasks."""
    print("\n" + "="*80)
    print("Transfer Learning with Multiple Source Tasks")
    print("="*80)
    
    setup_logging(level='INFO')
    
    config_space = create_spark_config_space()
    print(f"\n📋 Configuration space: {len(config_space.get_hyperparameters())} parameters")
    
    # ===============================================
    # Step 1: Generate historical data from source tasks
    # ===============================================
    print(f"\n📊 Step 1: Generating historical data from source tasks")
    print("-" * 80)
    
    source_histories = []
    source_workloads = ['sort', 'join', 'aggregate']
    
    for workload in source_workloads:
        history = generate_workload_history(config_space, workload, n_samples=60)
        source_histories.append(history)
    
    # ===============================================
    # Step 2: Calculate task similarities
    # ===============================================
    print(f"\n🎯 Step 2: Calculating similarities to target task (group-by)")
    print("-" * 80)
    
    target_workload = 'groupby'
    source_similarities = calculate_workload_similarities(target_workload)
    
    print(f"\nSource task similarities:")
    for idx, workload in enumerate(source_workloads):
        sim = source_similarities[idx]
        print(f"  - {workload:12s}: {sim:.3f} {'⭐' * int(sim * 5)}")
    
    # ===============================================
    # Step 3: Generate initial target task data
    # ===============================================
    print(f"\n📈 Step 3: Generating initial target task data")
    print("-" * 80)
    
    target_history = generate_workload_history(config_space, target_workload, n_samples=30)
    
    # ===============================================
    # Step 4: Multiple Source Tasks compression
    # ===============================================
    print(f"\n🔬 Step 4: Compression with multiple source tasks")
    print("-" * 80)
    
    compressor_multiple_source = Compressor(
        config_space=config_space,
        steps=[
            SHAPDimensionStep(topk=6),
            SHAPBoundaryRangeStep(top_ratio=0.75)
        ],
        save_compression_info=True,
        output_dir=f'{res_dir}/multiple_source'
    )
    
    all_histories = source_histories + [target_history]
    surrogate_space, sample_space = compressor_multiple_source.compress_space(
        space_history=all_histories,
        source_similarities=source_similarities
    )
    
    print(f"\n  Original dimensions: {len(config_space.get_hyperparameters())}")
    print(f"  Compressed dimensions: {len(surrogate_space.get_hyperparameters())}")
    print(f"  Compression ratio: {len(surrogate_space.get_hyperparameters())/len(config_space.get_hyperparameters()):.1%}")
    
    # ===============================================
    # Step 5: Single Source Task
    # ===============================================
    print(f"\n📊 Step 5: Single Source Task compression")
    print("-" * 80)
    
    compressor_single_source = Compressor(
        config_space=config_space,
        steps=[
            SHAPDimensionStep(topk=6),
            BoundaryRangeStep(method='boundary', top_ratio=0.75)
        ],
        save_compression_info=True,
        output_dir=f'{res_dir}/single_source'
    )
    
    surrogate_space_single_source, _ = compressor_single_source.compress_space(
        space_history=[target_history],
    )
    
    print(f"\n  Original dimensions: {len(config_space.get_hyperparameters())}")
    print(f"  Compressed dimensions: {len(surrogate_space_single_source.get_hyperparameters())}")
    
    # ===============================================
    # Step 6: Visualize and compare
    # ===============================================
    print(f"\n🎨 Step 6: Generating visualizations")
    print("-" * 80)
    
    # Visualize multiple source tasks results
    print("\n  Multiple Source Tasks compression:")
    visualize_compression_details(compressor_multiple_source, save_dir=f'{res_dir}/multiple_source/viz')
    
    # Visualize single source task results
    print("\n  Single Source Task compression:")
    visualize_compression_details(compressor_single_source, save_dir=f'{res_dir}/single_source/viz')
    
    
    # ===============================================
    # Summary
    # ===============================================
    print(f"\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"\n✅ Multiple Source Tasks and Single Source Task demonstration complete!")
    print(f"\n📊 Key results:")
    print(f"  - Source tasks: {len(source_histories)} (sort, join, aggregate)")
    print(f"  - Target task: group-by workload")
    print(f"\n🎨 Visualizations saved to:")
    print(f"  - {res_dir}/multiple_source/viz/")
    print(f"  - {res_dir}/single_source/viz/")
    
    return compressor_multiple_source, compressor_single_source


if __name__ == '__main__':
    main()

