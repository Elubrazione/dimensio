"""
Dimensio Comprehensive Examples

Demonstrates:
1. Creating configuration spaces
2. Generating mock history data
3. Using different compression step combinations
4. Visualizing compression effects
"""

import numpy as np
import os
from ConfigSpace import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)

from dimensio import (
    Compressor,
    SHAPDimensionStep,
    CorrelationDimensionStep,
    ExpertDimensionStep,
    BoundaryRangeStep,
    SHAPBoundaryRangeStep,
    KDEBoundaryRangeStep,
    ExpertRangeStep,
    QuantizationProjectionStep,
    REMBOProjectionStep,
    setup_logging,
    get_compressor
)
from dimensio.viz import visualize_compression_details
from utils import generate_mock_history

res_dir = "./results/comprehensive"


def create_sample_config_space():
    config_space = ConfigurationSpace(seed=42)
    
    config_space.add_hyperparameters([
        UniformFloatHyperparameter('spark.executor.memory', 1.0, 32.0, default_value=4.0),
        UniformFloatHyperparameter('spark.driver.memory', 1.0, 16.0, default_value=2.0),
        UniformFloatHyperparameter('spark.memory.fraction', 0.1, 0.9, default_value=0.6),
        UniformFloatHyperparameter('spark.memory.storageFraction', 0.1, 0.9, default_value=0.5),
        UniformFloatHyperparameter('spark.sql.shuffle.partitions.scale', 0.5, 2.0, default_value=1.0),
    ])
    
    config_space.add_hyperparameters([
        UniformIntegerHyperparameter('spark.executor.cores', 1, 16, default_value=4),
        UniformIntegerHyperparameter('spark.default.parallelism', 10, 1000, default_value=200),
        UniformIntegerHyperparameter('spark.sql.shuffle.partitions', 10, 2000, default_value=200),
        UniformIntegerHyperparameter('spark.reducer.maxSizeInFlight', 16, 256, default_value=48),
        UniformIntegerHyperparameter('spark.shuffle.file.buffer', 16, 256, default_value=32),
    ])
    
    config_space.add_hyperparameters([
        CategoricalHyperparameter('spark.serializer', 
                                 ['org.apache.spark.serializer.KryoSerializer',
                                  'org.apache.spark.serializer.JavaSerializer'],
                                 default_value='org.apache.spark.serializer.KryoSerializer'),
        CategoricalHyperparameter('spark.sql.adaptive.enabled', 
                                 ['true', 'false'],
                                 default_value='true'),
    ])
    
    print(f"✅ Created config space: {len(config_space.get_hyperparameters())} parameters")
    return config_space


def spark_objective(config_dict):
    score = 100.0
    
    if 'spark.executor.cores' in config_dict and 'spark.executor.memory' in config_dict:
        cores = config_dict['spark.executor.cores']
        memory = config_dict['spark.executor.memory']
        ratio_penalty = abs(memory / cores - 4.0) * 5
        score += ratio_penalty
    
    if 'spark.sql.shuffle.partitions' in config_dict:
        partitions = config_dict['spark.sql.shuffle.partitions']
        if partitions < 100:
            score += (100 - partitions) * 0.5
        elif partitions > 500:
            score += (partitions - 500) * 0.3
    
    if 'spark.memory.fraction' in config_dict:
        fraction = config_dict['spark.memory.fraction']
        score += abs(fraction - 0.65) * 50
    
    score += np.random.normal(0, 10)
    
    return max(score, 10.0)


def example_1_shap_dimension_range():
    """Example 1: SHAP dimension selection + range compression."""
    print("\n" + "="*80)
    print("Example 1: SHAP Dimension Selection + Boundary Range Compression")
    print("="*80)
    
    config_space = create_sample_config_space()
    
    history = generate_mock_history(config_space, n_samples=50, objective_func=spark_objective)
    
    # Define compression steps
    steps = [
        SHAPDimensionStep(strategy='shap', topk=6),
        BoundaryRangeStep(method='boundary', top_ratio=0.7, sigma=2.0)
    ]
    
    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir=f'{res_dir}/example1_shap_boundary'
    )
    
    # Perform compression
    surrogate_space, sample_space = compressor.compress_space(space_history=[history])
    
    # Print results
    print(f"\n📈 Compression results:")
    print(f"  - Original dims: {len(config_space.get_hyperparameters())}")
    print(f"  - Surrogate space: {len(surrogate_space.get_hyperparameters())}")
    print(f"  - Sample space: {len(sample_space.get_hyperparameters())}")
    
    summary = compressor.get_compression_summary()
    print(f"  - Compression ratio: {summary['surrogate_compression_ratio']:.2%}")
    
    # Visualization
    print(f"\n🎨 Generating visualizations...")
    visualize_compression_details(compressor, save_dir=f'{res_dir}/example1_shap_boundary/viz')
    
    print(f"✅ Example 1 complete! View results: {res_dir}/example1_shap_boundary/")
    
    return compressor


def example_2_correlation_shap_range():
    """Example 2: Correlation dimension selection + SHAP range compression."""
    print("\n" + "="*80)
    print("Example 2: Correlation Dimension Selection + SHAP Weighted Range Compression")
    print("="*80)
    
    config_space = create_sample_config_space()
    history = generate_mock_history(config_space, n_samples=50, objective_func=spark_objective)
    
    steps = [
        CorrelationDimensionStep(method='spearman', topk=5),
        SHAPBoundaryRangeStep(method='shap_boundary', top_ratio=0.8, sigma=1.5)
    ]
    
    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir=f'{res_dir}/example2_correlation_shap'
    )
    
    surrogate_space, sample_space = compressor.compress_space(space_history=[history])
    
    print(f"\n📈 Compressionresults:")
    print(f"  - Originaldimensions: {len(config_space.get_hyperparameters())}")
    print(f"  - Compressed dimensions: {len(surrogate_space.get_hyperparameters())}")
    
    visualize_compression_details(compressor, save_dir=f'{res_dir}/example2_correlation_shap/viz')
    
    print(f"✅ Example 2 complete！View results: {res_dir}/example2_correlation_shap/")
    
    return compressor


def example_3_kde_range():
    """Example 3: Pure KDE range compression (no dimensionality reduction)."""
    print("\n" + "="*80)
    print("Example 3: KDE Range Compression (retain all dimensions)")
    print("="*80)
    
    config_space = create_sample_config_space()
    history = generate_mock_history(config_space, n_samples=60, objective_func=spark_objective)
    
    steps = [
        KDEBoundaryRangeStep(
            method='kde_boundary',
            source_top_ratio=0.3,
            kde_coverage=0.6
        )
    ]
    
    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir=f'{res_dir}/example3_kde'
    )
    
    surrogate_space, sample_space = compressor.compress_space(space_history=[history])
    
    print(f"\n📈 Compressionresults:")
    print(f"  - Originaldimensions: {len(config_space.get_hyperparameters())} (unchanged)")
    print(f"  - rangeCompressionratio: {steps[0].kde_coverage:.0%}")
    
    visualize_compression_details(compressor, save_dir=f'{res_dir}/example3_kde/viz')
    
    print(f"✅ Example 3 complete！View results: {res_dir}/example3_kde/")
    
    return compressor


def example_4_quantization_projection():
    """Example 4: Quantization + REMBO projection."""
    print("\n" + "="*80)
    print("Example 4: Quantization Projection + REMBO Low-dimensional Embedding")
    print("="*80)
    
    config_space = create_sample_config_space()
    
    steps = [
        QuantizationProjectionStep(method='quantization', max_num_values=20),
        REMBOProjectionStep(method='rembo', low_dim=5)
    ]
    
    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir=f'{res_dir}/example4_quantization_rembo'
    )
    
    # Projection does not need history data
    surrogate_space, sample_space = compressor.compress_space()
    
    print(f"\n📈 Compressionresults:")
    print(f"  - Originaldimensions: {len(config_space.get_hyperparameters())}")
    print(f"  - Projected dimensions: {len(sample_space.get_hyperparameters())}")
    print(f"  - needunproject: {compressor.needs_unproject()}")
    
    print(f"\n🔄 Test sampling and unprojection:")
    sample_config = sample_space.sample_configuration()
    print(f"  - Sample space config: {list(sample_config.get_dictionary().keys())}")
    
    unprojected_config = compressor.unproject_point(sample_config)
    print(f"  - Unprojected config: {list(unprojected_config.get_dictionary().keys())}")
    
    visualize_compression_details(compressor, save_dir=f'{res_dir}/example4_quantization_rembo/viz')
    
    print(f"✅ Example 4 complete！View results: {res_dir}/example4_quantization_rembo/")
    
    return compressor


def example_5_expert_knowledge():
    """Example 5: Expert knowledge (dimension selection + range specification)."""
    print("\n" + "="*80)
    print("Example 5: Expert Knowledge-based Compression")
    print("="*80)
    
    config_space = create_sample_config_space()
    
    # Specify important parameters
    expert_params = [
        'spark.executor.memory',
        'spark.executor.cores',
        'spark.sql.shuffle.partitions',
        'spark.memory.fraction'
    ]
    # Specifyrange
    expert_ranges = {
        'spark.executor.memory': (2.0, 16.0),  # Limit to 2-16 GB
        'spark.executor.cores': (2, 8),  # Limit to 2-8 cores
        'spark.sql.shuffle.partitions': (100, 500),  # Limit to 100-500
        'spark.memory.fraction': (0.5, 0.8)  # Limit to 0.5-0.8
    }
    
    steps = [
        ExpertDimensionStep(strategy='expert', expert_params=expert_params),
        ExpertRangeStep(method='expert', expert_ranges=expert_ranges)
    ]
    
    compressor = Compressor(
        config_space=config_space,
        steps=steps,
        save_compression_info=True,
        output_dir=f'{res_dir}/example5_expert'
    )
    
    surrogate_space, sample_space = compressor.compress_space()
    
    print(f"\n📈 Compressionresults:")
    print(f"  - Originaldimensions: {len(config_space.get_hyperparameters())}")
    print(f"  - Expert selected dimensions: {len(expert_params)}")
    print(f"  - Compressed dimensions: {len(surrogate_space.get_hyperparameters())}")
    
    visualize_compression_details(compressor, save_dir=f'{res_dir}/example5_expert/viz')
    
    print(f"✅ Example 5 complete！View results: {res_dir}/example5_expert/")
    
    return compressor

def example_6_get_compressor_convenience():
    """Example 6: Using convenience function get_compressor."""
    print("\n" + "="*80)
    print("Example 6: Using Convenience Function to Create Compressor Quickly")
    print("="*80)
    
    config_space = create_sample_config_space()
    history = generate_mock_history(config_space, n_samples=50, objective_func=spark_objective)
    
    # Method 1: SHAP Strategy
    print(f"\n1️⃣ SHAP Strategy:")
    compressor_shap = get_compressor(
        compressor_type='shap',
        config_space=config_space,
        topk=5,
        top_ratio=0.8
    )
    surrogate_space, _ = compressor_shap.compress_space(space_history=[history])
    print(f"  - Compressed dimensions: {len(surrogate_space.get_hyperparameters())}")
    
    # Method 2: LlamaTune Strategy
    print(f"\n2️⃣ LlamaTune Strategy (Quantization + REMBO):")
    compressor_llama = get_compressor(
        compressor_type='llamatune',
        config_space=config_space,
        adapter_alias='rembo',
        le_low_dim=6,
        max_num_values=30
    )
    surrogate_space, _ = compressor_llama.compress_space()
    print(f"  - Projected dimensions: {len(surrogate_space.get_hyperparameters())}")
    
    # Method 3: Expert Strategy
    print(f"\n3️⃣ Expert Strategy:")
    compressor_expert = get_compressor(
        compressor_type='expert',
        config_space=config_space,
        expert_params=['spark.executor.memory', 'spark.executor.cores'],
        top_ratio=0.9
    )
    surrogate_space, _ = compressor_expert.compress_space(space_history=[history])
    print(f"  - Compressed dimensions: {len(surrogate_space.get_hyperparameters())}")
    
    print(f"\n✅ Example 6 complete！")


def main():
    """Run all examples."""
    
    setup_logging(level='INFO')
    
    os.makedirs(res_dir, exist_ok=True)
    
    try:
        # Run allExample
        example_1_shap_dimension_range()
        example_2_correlation_shap_range()
        example_3_kde_range()
        example_4_quantization_projection()
        example_5_expert_knowledge()
        example_6_get_compressor_convenience()
        
        print(f"\n📁 View results directory: {res_dir}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

