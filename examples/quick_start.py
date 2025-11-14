"""
Dimensio Quick Start Example

A simple example demonstrating basic usage.
"""

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter
from openbox.utils.history import History, Observation
from openbox.utils.constants import SUCCESS

from dimensio import (
    get_compressor,
    SHAPDimensionStep,
    BoundaryRangeStep,
    Compressor
)
from dimensio.viz import visualize_compression_details

res_dir = "./results/quick_start"

def create_simple_space():
    cs = ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        UniformFloatHyperparameter('learning_rate', 0.0001, 0.1, log=True),
        UniformFloatHyperparameter('momentum', 0.1, 0.99),
        UniformFloatHyperparameter('weight_decay', 0.0, 0.01),
        UniformIntegerHyperparameter('batch_size', 16, 512, log=True),
        UniformIntegerHyperparameter('num_layers', 1, 5),
        UniformIntegerHyperparameter('hidden_size', 64, 1024, log=True),
    ])
    return cs


def generate_simple_history(config_space, n_samples=30):
    history = History(
        task_id='quick_start',
        num_objectives=1,
        num_constraints=0,
        config_space=config_space
    )
    
    def objective(config_dict):
        # Mock objective function: learning_rate and batch_size are most important
        lr = config_dict.get('learning_rate', 0.01)
        bs = config_dict.get('batch_size', 32)
        
        # Optimal learning_rate ~= 0.001, batch_size ~= 128
        score = abs(np.log10(lr) + 3.0) * 10  # Optimal: 0.001
        score += abs(bs - 128) * 0.1  # Optimal: 128
        score += np.random.normal(0, 2)  # Add noise
        return max(score, 1.0)
    
    # Generate samples
    for _ in range(n_samples):
        config = config_space.sample_configuration()
        obj_value = objective(config.get_dictionary())
        obs = Observation(
            config=config,
            objectives=[obj_value],
            constraints=None,
            trial_state=SUCCESS,
            elapsed_time=0.1
        )
        history.update_observation(obs)
    
    return history


def main():
    print("="*60)
    print(" "*15 + "Dimensio Quick Start")
    print("="*60)
    
    print("\n📝 Step 1: Create configuration space")
    config_space = create_simple_space()
    print(f"   Created {len(config_space.get_hyperparameters())} hyperparameters")
    
    print("\n📊 Step 2: Generate mock history data")
    history = generate_simple_history(config_space, n_samples=30)
    objectives = [obs.objectives[0] for obs in history.observations]
    print(f"   Generated {len(history.observations)} evaluation samples")
    print(f"   Best: {min(objectives):.2f}, Mean: {np.mean(objectives):.2f}")
    
    print("\n🔧 Step 3: Create compressor using convenience function")
    compressor = get_compressor(
        compressor_type='shap',
        config_space=config_space,
        topk=3,  # Select top-3 important parameters
        top_ratio=0.8  # Use top-80% configs to compute range
    )
    
    print("\n⚙️  Step 4: Compress configuration space")
    surrogate_space, sample_space = compressor.compress_space(
        space_history=[history]
    )
    
    print(f"\n📈 Compression results:")
    print(f"   Original space: {len(config_space.get_hyperparameters())} dims")
    print(f"   Compressed:     {len(surrogate_space.get_hyperparameters())} dims")
    print(f"   Ratio:          {len(surrogate_space.get_hyperparameters())/len(config_space.get_hyperparameters()):.1%}")
    print(f"\n   Retained params: {surrogate_space.get_hyperparameter_names()}")
    
    print("\n🎲 Step 5: Sample new configs from compressed space")
    sampling_strategy = compressor.get_sampling_strategy()
    new_configs = sampling_strategy.sample(n=3)
    
    for i, config in enumerate(new_configs, 1):
        print(f"   Sample {i}: {config.get_dictionary()}")
    
    print("\n\n" + "="*60)
    print("Advanced: Custom compression step combination")
    print("="*60)
    
    steps = [
        SHAPDimensionStep(strategy='shap', topk=4),
        BoundaryRangeStep(method='boundary', top_ratio=0.7, sigma=2.0)
    ]
    
    compressor_custom = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir='./results/quick_start'
    )
    
    surrogate_space_custom, _ = compressor_custom.compress_space(
        space_history=[history]
    )
    
    print(f"\n📈 Custom compression results:")
    print(f"   Compressed dims: {len(surrogate_space_custom.get_hyperparameters())}")
    
    print("\n🎨 Step 6: Generate visualizations")
    visualize_compression_details(
        compressor_custom,
        save_dir=f'{res_dir}/viz'
    )
    print(f"   ✓ Visualizations saved to: {res_dir}/viz/")
    
    print(f"\n✅ Done! Check {res_dir}/ for detailed results")
    print("="*60)


if __name__ == '__main__':
    main()

