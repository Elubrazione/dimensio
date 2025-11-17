"""
Adaptive Update Strategies Comparison Example

Demonstrates different update strategies for adaptive dimension selection:
1. Periodic Update Strategy
2. Stagnation Detection Strategy
3. Improvement Detection Strategy
4. Composite Strategy (Stagnation + Improvement)
"""
import os

from dimensio import (
    Compressor,
    AdaptiveDimensionStep,
    BoundaryRangeStep,
    setup_logging
)
from dimensio.core.update import (
    PeriodicUpdateStrategy,
    StagnationUpdateStrategy,
    ImprovementUpdateStrategy,
    CompositeUpdateStrategy
)
from dimensio.steps.dimension import SHAPImportanceCalculator
from dimensio.viz import visualize_compression_details
from utils import create_simple_config_space, generate_history, generate_improving_history

res_dir = "./results/adaptive_strategies"

def run_adaptive_with_strategy(config_space, strategy, strategy_name, n_iterations=15):
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*80}")
    
    history = generate_history(config_space, n_samples=50, task_id=strategy_name)
    
    steps = [
        AdaptiveDimensionStep(
            importance_calculator=SHAPImportanceCalculator(),
            update_strategy=strategy,
            initial_topk=10,
            reduction_ratio=0.2,
            min_dimensions=4,
            max_dimensions=12
        ),
        BoundaryRangeStep(method='boundary', top_ratio=0.75)
    ]
    
    strategy_folder = strategy_name.split()[0].lower()
    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir=f'{res_dir}/{strategy_folder}'
    )
    
    surrogate_space, _ = compressor.compress_space(space_history=[history])
    
    print(f"Initial compression: {len(surrogate_space.get_hyperparameters())} dimensions")
    
    print(f"\n🔄 Running {n_iterations} iterations...")
    
    is_pure_improvement = 'Improvement Detection' == strategy_name
    is_composite = 'Composite' in strategy_name
    
    current_best = history.get_incumbent_value() if history.get_incumbent_value() is not None else 200.0
    
    for i in range(n_iterations):
        if is_pure_improvement:
            new_history = generate_improving_history(
                config_space, n_samples=5, iteration=i,
                task_id=f'{strategy_name}_{i}', verbose=False
            )
            best_in_new = min([obs.objectives[0] for obs in new_history.observations])
            if best_in_new >= current_best:
                improvement = 2.0 + i * 0.5
                new_history.observations[0].objectives[0] = max(current_best - improvement, 10.0)
            current_best = min(current_best, min([obs.objectives[0] for obs in new_history.observations]))
            
        elif is_composite:
            # Composite: alternating pattern
            # Iterations 0-4: improvement phase
            # Iterations 5-9: stagnation phase (random)
            # Iterations 10-14: improvement phase again
            if i < 5 or i >= 10:
                # Improvement phase
                new_history = generate_improving_history(
                    config_space, n_samples=5, iteration=i // 2,
                    task_id=f'{strategy_name}_{i}', verbose=False
                )
                # Ensure improvement
                best_in_new = min([obs.objectives[0] for obs in new_history.observations])
                if best_in_new >= current_best:
                    improvement = 2.0 + (i % 5) * 0.5
                    new_history.observations[0].objectives[0] = max(current_best - improvement, 10.0)
                current_best = min(current_best, min([obs.objectives[0] for obs in new_history.observations]))
            else:
                # Stagnation phase: generate random data (likely no improvement)
                new_history = generate_history(config_space, n_samples=5, 
                                            task_id=f'{strategy_name}_{i}', verbose=False)
        else:
            # Periodic and Stagnation: random data
            new_history = generate_history(config_space, n_samples=5, 
                                        task_id=f'{strategy_name}_{i}', verbose=False)
        
        for obs in new_history.observations:
            history.update_observation(obs)
        
        updated = compressor.update_compression(history)
        current_dims = len(compressor.surrogate_space.get_hyperparameters())
        
        if updated:
            if is_pure_improvement or is_composite:
                print(f"  ✓ Iter {i+1:2d}: Updated to {current_dims} dims (best: {current_best:.2f})")
            else:
                print(f"  ✓ Iter {i+1:2d}: Updated to {current_dims} dims")
    
    dimension_history = compressor._dimension_history
    print(f"\nFinal dimensions: {dimension_history[-1]}")
    print(f"Total updates: {sum(1 for i in range(1, len(dimension_history)) if dimension_history[i] != dimension_history[i-1])}")
    
    output_dir = f'{res_dir}/{strategy_folder}/viz'
    os.makedirs(output_dir, exist_ok=True)
    
    visualize_compression_details(compressor, save_dir=output_dir)
    
    iterations = compressor._iteration_history
    dimension_history = compressor._dimension_history
    return iterations, dimension_history


def compare_all_strategies():
    setup_logging(level='INFO')
    
    config_space = create_simple_config_space(n_float=8, n_int=7)
    print(f"✅ Created config space: {len(config_space.get_hyperparameters())} parameters")
    
    strategies = [
        (PeriodicUpdateStrategy(period=3), "Periodic (every 3 iters)"),
        (StagnationUpdateStrategy(threshold=3), "Stagnation Detection"),
        (ImprovementUpdateStrategy(threshold=2), "Improvement Detection"),
        (CompositeUpdateStrategy(
            StagnationUpdateStrategy(threshold=3),
            ImprovementUpdateStrategy(threshold=2)
        ), "Composite (Stagnation + Improvement)")
    ]
    
    # Run each strategy
    results = []
    for strategy, name in strategies:
        iters, dims = run_adaptive_with_strategy(config_space, strategy, name, n_iterations=15)
        results.append((name, iters, dims))
    
    print(f"\n📊 Creating comparison plot...")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['steelblue', 'coral', 'green', 'purple']
    
    for idx, (name, iters, dims) in enumerate(results):
        ax = axes[idx]
        
        ax.plot(iters, dims, marker='o', linewidth=2, markersize=6, 
               color=colors[idx], label=name)
        
        for i in range(1, len(dims)):
            if dims[i] != dims[i-1]:
                ax.axvline(x=iters[i], color='red', linestyle='--', alpha=0.3, linewidth=1)
                change = dims[i] - dims[i-1]
                change_str = f'+{change}' if change > 0 else str(change)
                ax.annotate(f'{dims[i]}\n({change_str})', 
                           xy=(iters[i], dims[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontweight='bold', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
        
        ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Dimensions', fontsize=11, fontweight='bold')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim([min(dims) - 1, max(dims) + 1])
    
    plt.tight_layout()
    plt.savefig(f'{res_dir}/adaptive_strategies_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {res_dir}/adaptive_strategies_comparison.png")
        
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    for name, iters, dims in results:
        updates = sum(1 for i in range(1, len(dims)) if dims[i] != dims[i-1])
        print(f"{name:40s}: {dims[0]:2d} → {dims[-1]:2d} dims ({updates} updates)")
    
    print(f"\n✅ All strategies completed!")
    print(f"📁 Results saved to {res_dir}")


if __name__ == '__main__':
    compare_all_strategies()

