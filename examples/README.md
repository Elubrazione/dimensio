# Dimensio Examples

This directory contains usage examples for Dimensio, demonstrating various compression strategies and visualization features.

## 📦 Utility Module

### `utils.py` - Common Utility Functions

Provides commonly used configuration space creation and data generation functions:

**Configuration Space Creation**:
- `create_simple_config_space(n_float, n_int)` - Create a simple parameter space
- `create_spark_config_space()` - Create a Spark configuration space

**Objective Functions**:
- `simple_objective(config_dict)` - Simple objective function
- `spark_objective(config_dict)` - Spark configuration objective function

**Data Generation**:
- `generate_history(config_space, n_samples, ...)` - Generate mock historical data
- `generate_mock_history(config_space, n_samples, ...)` - Generate mock historical data (alias)
- `generate_improving_history(config_space, iteration, ...)` - Generate data with improvement trends

**Usage Example**:
```python
from examples.utils import create_simple_config_space, generate_history

config_space = create_simple_config_space(n_float=10, n_int=5)
history = generate_history(config_space, n_samples=50)
```

## 📚 Example List

### 1. Quick Start (`quick_start.py`)

**Suitable for**: Beginners  
**Content**: 
- Creating simple configuration spaces
- Generating mock historical data
- Using convenience functions and custom steps
- Basic visualization

**Run**:
```bash
cd examples
python quick_start.py
```

**Output**: `./results/quick_start/`

**Key Code Snippets**:
```python
from dimensio import get_compressor, SHAPDimensionStep, BoundaryRangeStep

# Method 1: Using convenience function
compressor = get_compressor(
    compressor_type='shap',
    config_space=config_space,
    topk=3,
    top_ratio=0.8
)

# Method 2: Custom step combination
steps = [
    SHAPDimensionStep(strategy='shap', topk=4),
    BoundaryRangeStep(method='boundary', top_ratio=0.7, sigma=2.0)
]
compressor = Compressor(config_space=config_space, steps=steps)
```

---

### 2. Comprehensive Examples (`comprehensive.py`)

**Suitable for**: Users who want to understand all features  
**Content**: 6 complete examples covering:

#### Example 1: SHAP Dimension Selection + Standard Boundary Range Compression
- **Strategy**: `SHAPDimensionStep` + `BoundaryRangeStep`
- **Use case**: Data-driven parameter selection and range optimization
- **Output**: `./results/comprehensive/example1_shap_boundary/`

#### Example 2: Correlation Dimension Selection + SHAP Range Compression
- **Strategy**: `CorrelationDimensionStep` + `SHAPBoundaryRangeStep`
- **Use case**: Fast correlation analysis + importance-weighted range compression
- **Output**: `./results/comprehensive/example2_correlation_shap/`

#### Example 3: KDE Range Compression
- **Strategy**: `KDEBoundaryRangeStep` (range compression only, retain all dimensions)
- **Use case**: Density estimation-based range optimization
- **Output**: `./results/comprehensive/example3_kde/`

#### Example 4: Quantization + REMBO Projection
- **Strategy**: `QuantizationProjectionStep` + `REMBOProjectionStep`
- **Use case**: High-dimensional space dimensionality reduction, LlamaTune style
- **Output**: `./results/comprehensive/example4_quantization_rembo/`

#### Example 5: Expert Knowledge
- **Strategy**: `ExpertDimensionStep` + `ExpertRangeStep`
- **Use case**: Manual configuration based on domain knowledge
- **Output**: `./results/comprehensive/example5_expert/`

#### Example 6: Convenience Functions
- **Strategy**: Using `get_compressor()` for quick creation
- **Demonstrates**: Three preset strategies: SHAP, LlamaTune, Expert

**Run**:
```bash
cd examples
python comprehensive.py
```

**Output**: `./results/comprehensive/example{1-6}_*/`

---

### 3. Adaptive Update Strategies Comparison (`adaptive_strategies.py`)

**Suitable for**: Users who want to deeply understand adaptive update strategies  
**Content**: Compares 4 different update strategy effects

**Strategy Description**:
- **Periodic Update** (Periodic): Automatically updates every 3 iterations
- **Stagnation Detection** (Stagnation): Triggers when no improvement for 3 consecutive iterations
- **Improvement Detection** (Improvement): Triggers on 2 consecutive improvements  
- **Composite Strategy** (Composite): **Mixed data pattern**
  - Iterations 0-4: Improvement phase → triggers Improvement
  - Iterations 5-9: Stagnation phase → triggers Stagnation
  - Iterations 10-14: Improvement again → triggers Improvement

#### 4 Update Strategies:

1. **Periodic Strategy**
   - Automatically triggers update every N iterations
   - Use case: Regular adjustments, independent of performance changes
   - Parameter: `period=3` (triggers every 3 iterations)

2. **Stagnation Detection**
   - Triggers when performance has no improvement for N consecutive iterations
   - Use case: Increase dimensions when performance stagnates to explore more space
   - Parameter: `threshold=3` (3 consecutive stagnations)
   - Behavior: Increase dimensions

3. **Improvement Detection**
   - Triggers when performance improves for N consecutive iterations
   - Use case: Reduce dimensions when performance is good to improve efficiency
   - Parameter: `threshold=2` (2 consecutive improvements)
   - Behavior: Decrease dimensions

4. **Composite Strategy**
   - Combines multiple strategies, updates when any strategy triggers
   - Use case: Balance exploration and exploitation
   - Example: `Stagnation + Improvement`

**Run**:
```bash
cd examples
python adaptive_strategies.py
```

**Output**: 
- `./results/adaptive_strategies/periodic/` - Periodic strategy results
- `./results/adaptive_strategies/stagnation/` - Stagnation detection strategy results
- `./results/adaptive_strategies/improvement/` - Improvement detection strategy results
- `./results/adaptive_strategies/composite/` - Composite strategy results
- `./results/adaptive_strategies/adaptive_strategies_comparison.png` - Four-panel comparison chart

**Key Code Snippets**:
```python
from dimensio import AdaptiveDimensionStep
from dimensio.core.update import (
    PeriodicUpdateStrategy,
    StagnationUpdateStrategy,
    ImprovementUpdateStrategy,
    CompositeUpdateStrategy
)

# Periodic strategy
step = AdaptiveDimensionStep(
    update_strategy=PeriodicUpdateStrategy(period=3),
    initial_topk=10,
    reduction_ratio=0.2,
    min_dimensions=4,
    max_dimensions=12
)

# Composite strategy
step = AdaptiveDimensionStep(
    update_strategy=CompositeUpdateStrategy(
        StagnationUpdateStrategy(threshold=3),
        ImprovementUpdateStrategy(threshold=2)
    ),
    initial_topk=10,
    reduction_ratio=0.2,
    min_dimensions=4,
    max_dimensions=12
)
```

---

### 4. Multi-Source Task Transfer Learning (`multi_single_source.py`)

**Suitable for**: Users who want to leverage historical task data to accelerate new task optimization  
**Content**: Demonstrates how to use historical data from multiple source tasks to optimize new target tasks

#### Scenario Description:
Simulates Spark configuration optimization for different workload types:
- **Source Task 1**: Sort workload (past optimization data)
- **Source Task 2**: Join workload (past optimization data)
- **Source Task 3**: Aggregate workload (past optimization data)
- **Target Task**: Group-by workload (new task to optimize)

#### Core Features:
1. **Task Similarity Calculation** - Calculate task similarities based on domain knowledge or meta-features
2. **Weighted Transfer Learning** - Weight source task data according to similarity
3. **Similarity Visualization** - Automatically generate task similarity heatmap


**Run**:
```bash
cd examples
python multi_single_source.py
```

**Output**: 
- `./results/multiple_single_source/multiple_source/` - Compression results using multi-source task transfer learning
- `./results/multiple_single_source/single_source/` - Results using current task data
- Includes `source_task_similarities.png` - Source task similarity visualization
- Includes `multi_task_importance_heatmap.png` - Multi-task parameter importance heatmap

**Key Code Snippets**:
```python
from dimensio import SHAPDimensionStep, SHAPBoundaryRangeStep

# Calculate task similarities
source_similarities = {
    0: 0.65,  # Sort workload similarity
    1: 0.80,  # Join workload similarity
    2: 0.75   # Aggregate workload similarity
}

# Multi-source task compression
compressor = Compressor(
    config_space=config_space,
    steps=[
        SHAPDimensionStep(topk=6),
        SHAPBoundaryRangeStep(top_ratio=0.75)
    ]
)

# Compress using multi-source historical data
surrogate_space, sample_space = compressor.compress_space(
    space_history=source_histories + [target_history],
    source_similarities=source_similarities
)
```

## 📊 Output Files

Each example also generates:

- **compression_initial_compression_*.json** - Initial compression details
- **compression_history.json** - Compression history records (if updates occurred)
- Contains detailed information and compression statistics for all steps


## 📖 More Information

- [Main Documentation](../README.md)
- [Chinese Documentation](../README_CN.md) ([Chinese Examples](./README_CN.md))
- [API Documentation](../docs/)
- [Publishing Guide](../docs/PUBLISH_GUIDE.md)

## 📝 Contributing Examples

If you want to contribute new examples:

1. Create a new Python file in the `examples/` directory
2. Use utility functions from `utils.py`
3. Add detailed docstrings
4. Ensure examples can run independently
5. Update the current README.md file

## 📧 Feedback

If you encounter issues or have suggestions for improvement when using the examples, please:
- Submit an Issue: [GitHub Issues](https://github.com/Elubrazione/dimensio/issues)
- Send an email: lingchingtung@stu.pku.edu.cn

