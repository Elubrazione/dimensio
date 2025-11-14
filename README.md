# Dimensio

English | [简体中文](./README_CN.md)

A flexible configuration space compression library designed for Bayesian Optimization. Supports combining multiple compression strategies through a Pipeline architecture to significantly improve the efficiency of high-dimensional hyperparameter optimization.

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Features](#features)
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Compression Strategies](#compression-strategies)
- [Visualization](#visualization)
- [Integration with Bayesian Optimization](#integration-with-bayesian-optimization)
- [API Documentation](#api-documentation)
- [Advanced Usage](#advanced-usage)

## Features

✨ **Pipeline Architecture**: Build flexible compression strategies by combining multiple compression steps  
🎯 **Multiple Compression Strategies**: Support dimension selection, range compression, and projection transformations  
🔄 **Adaptive Updates**: Dynamically adjust compression strategies during optimization  
🎨 **Rich Visualizations**: Provide various visualization tools for compression process and parameter importance  
📊 **Transfer Learning Support**: Dynamically transform multi-source historical data to adapt to compression strategy changes  
🔧 **Extensible Design**: Easy to add custom compression steps and filling strategies

## Overview

### Space Concepts

**Original Space**
- Complete, uncompressed configuration space
- Contains all parameters with their original ranges

**Sample Space**
- Space used for sampling new configurations
- Affected by dimension selection and range compression
- Low-dimensional if projection steps are used

**Surrogate Space**
- Space used for surrogate model training and prediction
- Final output space of the pipeline

**Unprojected Space**
- Space before projection
- Used to map low-dimensional configs back to high-dimensional space for **evaluation**
  - If dimension compression or range compression was performed before the projection step, this space is the space after dimension/range compression and before projection; otherwise, it is the original space.

### Compression Flow

```
Original Space
    ↓ [Dimension Selection - DimensionSelectionStep]
Dimension-reduced Space
    ↓ [Range Compression - RangeCompressionStep]
Range-compressed Space
    ↓ [Projection - ProjectionStep]
Final Returned Compressed Space
    ├── Sample Space: for generating new configurations
    └── Surrogate Space: for model training
```

## Installation

### From PyPI

```bash
pip install dimensio
```

### From Source

```bash
git clone https://github.com/Elubrazione/dimensio.git
cd dimensio
pip install -e .
```

## Quick Start

> 💡 **See Full Examples**: The [examples/](./examples/) directory contains multiple runnable examples covering all features and use cases. See [examples/README.md](./examples/README.md) for detailed documentation.

### Basic Usage
**Strongly recommend combining compression steps `step` yourself**

```python
from dimensio import Compressor, SHAPDimensionStep, BoundaryRangeStep
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

# 1. Create configuration space
config_space = ConfigurationSpace()
config_space.add_hyperparameter(UniformFloatHyperparameter('x1', 1, 100))
config_space.add_hyperparameter(UniformFloatHyperparameter('x2', -5, 1028))
config_space.add_hyperparameter(UniformFloatHyperparameter('x3', 3140, 7890))

# 2. Define compression steps
steps = [
    SHAPDimensionStep(strategy='shap', topk=2),
    BoundaryRangeStep(method='boundary', top_ratio=0.8)
]

# 3. Create compressor
compressor = Compressor(
    config_space=config_space,
    steps=steps,
    save_compression_info=True,
    output_dir='./results/compression'
)

# 4. Compress configuration space
surrogate_space, sample_space = compressor.compress_space(space_history=None)

print(f"Original dimensions: {len(config_space.get_hyperparameters())}")
print(f"Surrogate space dimensions: {len(surrogate_space.get_hyperparameters())}")
print(f"Sample space dimensions: {len(sample_space.get_hyperparameters())}")
```

### Using Convenience Functions

```python
from dimensio import get_compressor

# LlamaTune strategy (quantization + projection)
compressor = get_compressor(
    compressor_type='llamatune',
    config_space=config_space,
    adapter_alias='rembo',  # or 'hesbo'
    le_low_dim=10,
    max_num_values=50
)

# Expert knowledge strategy
compressor = get_compressor(
    compressor_type='expert',
    config_space=config_space,
    expert_params=['x1', 'x3'],
    top_ratio=0.9
)
```

### Configure Logging

```python
from dimensio import setup_logging, disable_logging
import logging

# Set log level
setup_logging(level=logging.INFO)

# Or save to file
setup_logging(level=logging.DEBUG, log_file='dimensio.log')

# Disable logging
disable_logging()
```

## Examples

The `examples/` directory contains comprehensive, runnable examples:

### 1. Quick Start (`quick_start.py`)
A simple example demonstrating basic usage:
- Creating configuration spaces
- Generating mock history data
- Using convenience functions and custom steps
- Basic visualization

**Run**: `python examples/quick_start.py`

### 2. Comprehensive Examples (`comprehensive.py`)
Six complete examples covering different compression strategies:
- **Example 1**: SHAP dimension selection + standard boundary range compression
- **Example 2**: Correlation dimension selection + SHAP range compression
- **Example 3**: KDE range compression (retain all dimensions)
- **Example 4**: Quantization + REMBO projection
- **Example 5**: Expert knowledge-based compression
- **Example 6**: Using convenience functions

**Run**: `python examples/comprehensive.py`

### 3. Adaptive Update Strategies (`adaptive_strategies.py`)
Compares four different adaptive update strategies:
- **Periodic Update**: Updates at fixed intervals
- **Stagnation Detection**: Triggers when optimization stagnates
- **Improvement Detection**: Triggers on consecutive improvements
- **Composite Strategy**: Combines multiple strategies (demonstrates Stagnation + Improvement)

**Run**: `python examples/adaptive_strategies.py`

### 4. Multi-Source Transfer Learning (`multi_single_source.py`)
Demonstrates transfer learning with multiple source tasks:
- Generating historical data from multiple source tasks
- Calculating task similarities
- Comparing single-source vs multi-source compression
- Visualizing transfer learning effects

**Run**: `python examples/multi_single_source.py`

For detailed documentation on all examples, see [examples/README.md](./examples/README.md).

## Compression Strategies

### 1. Dimension Selection

Reduce the number of parameters by keeping the most important ones.

#### SHAPDimensionStep

Parameter selection based on SHAP values. **Supports multi-source transfer learning**.

```python
from dimensio import SHAPDimensionStep

step = SHAPDimensionStep(
    strategy='shap',
    topk=10  # Select top-10 important parameters
)
```

**How it works**:
1. Train a Random Forest regression model using historical evaluation data
2. Calculate SHAP values to quantify each parameter's importance
3. Select the top-k most important parameters

**Transfer learning support**:
- Can leverage historical data from multiple source tasks
- Automatically weight different sources by task similarity

#### CorrelationDimensionStep

Parameter selection based on Spearman or Pearson correlation. **Supports multi-source transfer learning**.

```python
from dimensio import CorrelationDimensionStep

step = CorrelationDimensionStep(
    method='spearman',  # or 'pearson'
    topk=10
)
```

**How it works**:
- Calculate correlation (Spearman or Pearson) between each parameter and objective
- Select parameters with highest correlation

**Transfer learning support**:
- Can leverage historical data from multiple source tasks
- Automatically weight different sources by task similarity

#### ExpertDimensionStep

Parameter selection based on expert knowledge.

```python
from dimensio import ExpertDimensionStep

step = ExpertDimensionStep(
    strategy='expert',
    expert_params=['param1', 'param2', 'param3']
)
```

#### AdaptiveDimensionStep

Adaptively adjust the number of parameters. Configurable importance calculator and update strategy.

```python
from dimensio import AdaptiveDimensionStep
from dimensio.steps.dimension import SHAPImportanceCalculator
from dimensio.core.update import PeriodicUpdateStrategy

step = AdaptiveDimensionStep(
    importance_calculator=SHAPImportanceCalculator(),  # Optional, default SHAP
    update_strategy=PeriodicUpdateStrategy(period=5),  # Update every 5 iterations
    initial_topk=30,
    reduction_ratio=0.2,
    min_dimensions=5,
    max_dimensions=50  # Optional
)
```

**Parameters**:
- `importance_calculator`: Importance calculator (default SHAP)
- `update_strategy`: Update strategy (default every 5 iterations), see below for options
- `initial_topk`: Initial number of parameters
- `reduction_ratio`: Ratio for dimension adjustment per update (for increase or decrease)
- `min_dimensions`: Minimum number of dimensions
- `max_dimensions`: Maximum number of dimensions (optional)

**Supported Update Strategies**:

##### 1. PeriodicUpdateStrategy (Periodic Update)

Execute updates at fixed iteration intervals, gradually reducing parameter count.

```python
from dimensio.core.update import PeriodicUpdateStrategy

update_strategy = PeriodicUpdateStrategy(period=10)  # Update every 10 iterations
```

**Behavior**: Every `period` iterations, reduce by `current_topk × reduction_ratio` parameters.

##### 2. StagnationUpdateStrategy (Stagnation Detection)

Increase parameter count when optimization stagnates to expand search space.

```python
from dimensio.core.update import StagnationUpdateStrategy

update_strategy = StagnationUpdateStrategy(threshold=5)  # Trigger after 5 stagnant iterations
```

**Behavior**: When best value hasn't improved for `threshold` consecutive iterations, increase by `current_topk × reduction_ratio` parameters.

##### 3. ImprovementUpdateStrategy (Improvement Detection)

Reduce parameter count when improvements are detected to focus search.

```python
from dimensio.core.update import ImprovementUpdateStrategy

update_strategy = ImprovementUpdateStrategy(threshold=3)  # Trigger after 3 consecutive improvements
```

**Behavior**: When best value improves for `threshold` consecutive iterations, reduce by `current_topk × reduction_ratio` parameters.

##### 4. HybridUpdateStrategy (Hybrid Strategy)

Combines periodic, stagnation detection, and improvement detection strategies.

```python
from dimensio.core.update import HybridUpdateStrategy

update_strategy = HybridUpdateStrategy(
    period=10,                    # Base period: every 10 iterations
    stagnation_threshold=5,       # Stagnation: 5 iterations without improvement
    improvement_threshold=3       # Improvement: 3 consecutive improvements
)
```

**Behavior**:
- **Priority**: Stagnation > Improvement > Periodic
- On stagnation: increase dimensions
- On improvement: reduce dimensions
- On period reached: reduce dimensions

##### 5. CompositeUpdateStrategy (Composite Strategy)

Freely combine multiple strategies, update when any strategy triggers.

```python
from dimensio.core.update import CompositeUpdateStrategy, StagnationUpdateStrategy, ImprovementUpdateStrategy

update_strategy = CompositeUpdateStrategy(
    StagnationUpdateStrategy(threshold=5),
    ImprovementUpdateStrategy(threshold=3)
)
```

**Behavior**: Check each strategy in order, the first triggered strategy determines how to update dimensions.

**Usage Recommendations**:
- **Stable optimization**: Use `PeriodicUpdateStrategy` for smooth dimension reduction
- **Prone to stagnation**: Use `StagnationUpdateStrategy` or `HybridUpdateStrategy`
- **Fast convergence**: Use `ImprovementUpdateStrategy`
- **Complex scenarios**: Use `HybridUpdateStrategy` or `CompositeUpdateStrategy`

### 2. Range Compression

Narrow parameter value ranges to focus on high-value regions.

#### BoundaryRangeStep

Range compression based on mean and standard deviation of best configurations.

```python
from dimensio import BoundaryRangeStep

step = BoundaryRangeStep(
    method='boundary',
    top_ratio=0.8,  # Use top-80% configs to compute bounds
    sigma=2.0       # Standard deviation multiplier (μ ± 2σ)
)
```

#### SHAPBoundaryRangeStep

SHAP-weighted range compression. **Supports multi-source transfer learning**.

```python
from dimensio import SHAPBoundaryRangeStep

step = SHAPBoundaryRangeStep(
    method='shap_boundary',
    top_ratio=0.8,
    sigma=2.0
)
```

**How it works**:
- Adjust compression level based on parameter importance
- Important parameters retain larger search ranges

#### KDEBoundaryRangeStep

Range compression based on kernel density estimation. **Supports multi-source transfer learning**.

```python
from dimensio import KDEBoundaryRangeStep

step = KDEBoundaryRangeStep(
    method='kde_boundary',
    source_top_ratio=0.3,  # Use top-30% configs from source tasks
    kde_coverage=0.6       # KDE coverage ratio (retain 60% of probability density)
)
```

**How it works**:
- Use KDE (Kernel Density Estimation) to estimate parameter probability density distribution
- Determine high-density region retention ratio based on `kde_coverage`
- For multi-source task data, weight by task similarity

#### ExpertRangeStep

Expert-specified parameter ranges.

```python
from dimensio import ExpertRangeStep

step = ExpertRangeStep(
    method='expert',
    expert_ranges={
        'param1': (0, 10),
        'param2': (5, 15)
    }
)
```

### 3. Projection

Transform parameter representation to reduce search complexity.

#### QuantizationProjectionStep

Integer parameter quantization, compressing large-range integer parameters to smaller discrete value sets.

```python
from dimensio import QuantizationProjectionStep

step = QuantizationProjectionStep(
    method='quantization',
    max_num_values=50,  # Maximum number of discrete values
    adaptive=False  # Whether to adaptively adjust
)
```

**How it works**:
- Only quantizes `UniformIntegerHyperparameter` with range larger than `max_num_values`
- Maps original range `[lower, upper]` to quantized range `[1, max_num_values]`
- Samples integers in quantized space, unprojects back to original range for evaluation
- Other parameter types remain unchanged

**Example**:
- Original parameter: `x ∈ [1000, 5000]` (4001 values)
- Quantized: `x|q ∈ [1, 50]` (50 values)
- Compression ratio: 50/4001 ≈ 1.25%

#### REMBOProjectionStep

Random Embedding Bayesian Optimization.

```python
from dimensio import REMBOProjectionStep

step = REMBOProjectionStep(
    method='rembo',
    low_dim=10,  # Low-dimensional space dimension
    max_num_values=50  # Use with quantization
)
```

**How it works**:
- Assumes high-dimensional space has low-dimensional effective subspace
- Project d dimensions to d_e dimensions via random matrix (d_e << d)
- Low-dim sampling range: `[-√d_e, √d_e]`

#### HesBOProjectionStep

Hashing Embedding Bayesian Optimization.

```python
from dimensio import HesBOProjectionStep

step = HesBOProjectionStep(
    method='hesbo',
    low_dim=10,
    max_num_values=50
)
```

**How it works**:
- Use hashing functions for dimension mapping
- Low-dim sampling range: `[-1, 1]`
- More memory-efficient than REMBO

#### KPCAProjectionStep

Kernel Principal Component Analysis projection.

```python
from dimensio import KPCAProjectionStep

step = KPCAProjectionStep(
    method='kpca',
    n_components=10,
    kernel='rbf'
)
```

**How it works**:
- Extract nonlinear principal components using kernel methods
- **Note:** This method only uses the extracted principal component dimensions to train the surrogate model; the returned sample space is still the space before this step

## Visualization

Dimensio provides rich visualization tools to analyze compression effects. The visualization system **automatically detects** which compression steps are used and generates relevant plots.

### Automatic Visualization

```python
from dimensio.viz import visualize_compression_details

# Automatically generates all relevant visualizations based on used steps
visualize_compression_details(
    compressor=compressor,
    save_dir='./results/visualization'
)
```

**Generated plots** (automatically selected based on compression steps):

1. **compression_summary.png**: Compression summary
   - Dimension changes across steps
   - Compression ratio statistics
   - Range compression statistics
   - Text summary

2. **range_compression_step_*.png**: Detailed view for each range compression step
   - ✅ Auto-detected when using `BoundaryRangeStep`, `SHAPBoundaryRangeStep`, `KDEBoundaryRangeStep`
   - Original range vs compressed range
   - Compression ratio for each parameter
   - Quantization info (if used)

3. **parameter_importance_step_*.png**: Parameter importance visualization
   - ✅ Auto-detected when using `SHAPDimensionStep`, `CorrelationDimensionStep`, `AdaptiveDimensionStep`
   - Top-K parameter importance scores

4. **dimension_evolution.png**: Dimension evolution curve
   - ✅ Auto-detected when using `AdaptiveDimensionStep` with update history
   - Shows dimension changes over iterations
   - Highlights each dimension adjustment

5. **source_task_similarities.png**: Source task similarities
   - ✅ Auto-detected when using multi-source task (≥2 tasks) transfer learning (providing `source_similarities`)
   - Bar chart of similarity scores between source tasks and target task

6. **multi_task_importance_heatmap_step_*.png**: Multi-task importance heatmap
   - ✅ Auto-detected when using SHAP/Correlation-based dimension compression methods + multiple source tasks
   - Heatmap comparing parameter importance across different tasks
   - Useful for discovering common important parameters and task-specific key parameters

### Manual Visualization

You can also call specific visualization functions:

```python
from dimensio.viz import visualize_parameter_importance, visualize_importance_heatmap

# Single-task parameter importance
visualize_parameter_importance(
    param_names=['x1', 'x2', 'x3', ...],
    importances=[0.5, 0.3, 0.2, ...],
    save_path='./results/parameter_importance.png',
    topk=20
)

# Multi-task importance heatmap
import numpy as np
importances = np.array([
    [0.5, 0.3, 0.2, ...],  # Task 1
    [0.4, 0.4, 0.2, ...],  # Task 2
    [0.6, 0.2, 0.2, ...]   # Task 3
])

visualize_importance_heatmap(
    param_names=['x1', 'x2', 'x3', ...],
    importances=importances,
    save_path='./results/importance_heatmap.png',
    tasks=['Task 1', 'Task 2', 'Task 3']
)
```

## Integration with Bayesian Optimization

Dimensio can be seamlessly integrated into Bayesian Optimization systems:
- Use compressor's transformation interface `transform_source_data` to transform historical data automatically
- Use `surrogate_space` to train surrogate models, use `sample_space` for data sampling
- If sampled configurations are projected, they can be converted via `compressor.unproject_point()`, and the converted space can be obtained via `compressor.get_unprojected_space()`

### Integration with Advisor

```python
from openbox import Advisor
from dimensio import get_compressor

# 1. Create compressor
compressor = get_compressor(
    compressor_type='shap',
    config_space=config_space,
    topk=10,
    top_ratio=0.8
)

# 2. Compress configuration space
surrogate_space, sample_space = compressor.compress_space()

# 3. Create Advisor (use compressed space)
advisor = Advisor(
    config_space=surrogate_space,
    num_objectives=1,
    num_constraints=0,
    # ... other parameters
)

# 4. Use in optimization loop
for iteration in range(max_iterations):
    # 4.1 Get suggested configs from sample space
    sampling_strategy = compressor.get_sampling_strategy()
    configs = sampling_strategy.sample(n=batch_size)
    
    # 4.2 Unproject if using projection
    if compressor.needs_unproject():
        eval_configs = []
        for config in configs:
            unprojected_dict = compressor.unproject_point(config)
            # Create config in original space for evaluation
            eval_config = Configuration(config_space, values=unprojected_dict)
            eval_configs.append(eval_config)
    else:
        eval_configs = configs
    
    # 4.3 Evaluate configs
    results = []
    for config in eval_configs:
        obj_value = objective_function(config)
        results.append((config, obj_value))
    
    # 4.4 Update Advisor
    for eval_config, obj_value in results:
        # Convert to surrogate space
        surrogate_config = compressor.convert_config_to_surrogate_space(eval_config)
        advisor.update_observation(
            observation=(surrogate_config, obj_value)
        )
    
    # 4.5 Adaptive update compression (optional)
    if iteration % update_interval == 0:
        updated = compressor.update_compression(advisor.history)
        if updated:
            # Update Advisor's config space
            advisor.config_space = compressor.surrogate_space
            # Transform history
            advisor.history = transform_history(
                advisor.history, 
                compressor.surrogate_space
            )
```

### Integration with Optimizer (with Transfer Learning)

```python
from openbox import Optimizer
from dimensio import get_compressor

class CompressedOptimizer:
    def __init__(self, config_space, compressor_config, **kwargs):
        # 1. Create compressor
        self.compressor = get_compressor(
            config_space=config_space,
            **compressor_config
        )
        
        # 2. Load source task histories
        source_hpo_data = self.load_source_histories()
        
        # 3. Compress configuration space
        surrogate_space, sample_space = self.compressor.compress_space(
            space_history=source_hpo_data
        )
        
        # 4. Transform source data to compressed space
        transformed_source_data = self.compressor.transform_source_data(
            source_hpo_data
        )
        
        # 5. Create Optimizer
        self.optimizer = Optimizer(
            config_space=surrogate_space,
            source_hpo_data=transformed_source_data,
            **kwargs
        )
    
    def optimize(self, objective_function, max_iterations):
        for iteration in range(max_iterations):
            # Get suggested config
            config = self.optimizer.ask()
            
            # Unproject if needed
            if self.compressor.needs_unproject():
                eval_dict = self.compressor.unproject_point(config)
                eval_config = Configuration(
                    self.compressor.origin_config_space,
                    values=eval_dict
                )
            else:
                eval_config = config
            
            # Evaluate
            obj_value = objective_function(eval_config)
            
            # Tell result (use compressed space config)
            self.optimizer.tell(config, obj_value)
            
            # Adaptive update
            if iteration % 10 == 0:
                self.compressor.update_compression(self.optimizer.history)
        
        return self.optimizer.get_incumbent()
```

### Complete Example

See integration examples in [multique_fidelity_spark](https://github.com/Elubrazione/multique_fidelity_spark) project:

```
multique_fidelity_spark/
├── Compressor/          # Early version of Dimensio
├── Advisor/             # Advisor with compressor integration
├── Optimizer/           # Optimizer implementation
└── main.py             # Complete usage example
```

## API Documentation

### Compressor

Main compressor class that manages compression pipeline and configuration space transformations.

```python
class Compressor:
    def __init__(
        self,
        config_space: ConfigurationSpace,
        steps: Optional[List[CompressionStep]] = None,
        filling_strategy: Optional[FillingStrategy] = None,
        save_compression_info: bool = False,
        output_dir: str = './results/compression',
        **kwargs
    ):
        """
        Args:
            config_space: Original configuration space
            steps: List of compression steps
            filling_strategy: Filling strategy (default: uses search space default values)
            save_compression_info: Whether to save compression info
            output_dir: Output directory
        """
```

**Main methods**:

```python
def compress_space(
    self,
    space_history: Optional[List] = None,
    source_similarities: Optional[Dict[int, float]] = None
) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
    """
    Compress configuration space
    
    Args:
        space_history: Historical data (for SHAP, KDE, etc.)
        source_similarities: Source task similarities (for transfer learning)
    
    Returns:
        (surrogate_space, sample_space)
    """

def convert_config_to_surrogate_space(
    self,
    config: Configuration
) -> Configuration:
    """Convert config to surrogate space"""

def unproject_point(self, point: Configuration) -> dict:
    """Unproject config (projection step -> original space)"""

def update_compression(self, history: History) -> bool:
    """Adaptive update of compression strategy"""

def get_sampling_strategy(self) -> SamplingStrategy:
    """Get sampling strategy"""

def transform_source_data(
    self,
    source_hpo_data: Optional[List[History]]
) -> Optional[List[History]]:
    """Transform source task data to current compressed space"""

def get_compression_summary(self) -> dict:
    """Get compression summary info"""
```

### CompressionStep

Base class for compression steps.

```python
class CompressionStep(ABC):
    @abstractmethod
    def compress(
        self,
        config_space: ConfigurationSpace,
        space_history: Optional[List] = None,
        source_similarities: Optional[Dict[int, float]] = None
    ) -> ConfigurationSpace:
        """Execute compression"""
    
    def affects_sampling_space(self) -> bool:
        """Whether affects sampling space"""
    
    def needs_unproject(self) -> bool:
        """Whether needs unprojection"""
    
    def supports_adaptive_update(self) -> bool:
        """Whether supports adaptive update"""
    
    def get_sampling_strategy(self) -> Optional[SamplingStrategy]:
        """Get sampling strategy"""
```

### SamplingStrategy

Sampling strategy interface.

```python
class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, n: int = 1) -> List[Configuration]:
        """Sample n configurations"""

# Standard sampling strategy
class StandardSamplingStrategy(SamplingStrategy):
    def __init__(self, config_space: ConfigurationSpace, seed: int = 42):
        ...
```

### FillingStrategy

Filling strategy interface for handling parameter filling during dimension changes.

```python
class FillingStrategy(ABC):
    @abstractmethod
    def fill_missing_parameters(
        self,
        config_dict: Dict[str, Any],
        target_space: ConfigurationSpace
    ) -> Dict[str, Any]:
        """Fill missing parameters"""

# Default filling (use search space default values)
class DefaultValueFilling(FillingStrategy):
    ...

# Clipping filling (clip to range)
class ClippingValueFilling(FillingStrategy):
    ...
```

### UpdateStrategy

Update strategy interface for adaptive updates in `AdaptiveDimensionStep`.

```python
class UpdateStrategy(ABC):
    @abstractmethod
    def should_update(self, progress: OptimizerProgress, history: History) -> bool:
        """Determine if update should be performed"""
    
    @abstractmethod
    def compute_new_topk(
        self,
        current_topk: int,
        reduction_ratio: float,
        min_dimensions: int,
        max_dimensions: Optional[int],
        progress: OptimizerProgress
    ) -> Tuple[int, str]:
        """Compute new parameter count"""

# Periodic update strategy
class PeriodicUpdateStrategy(UpdateStrategy):
    def __init__(self, period: int = 10):
        """period: Update period (number of iterations)"""

# Stagnation detection update strategy
class StagnationUpdateStrategy(UpdateStrategy):
    def __init__(self, threshold: int = 5):
        """threshold: Stagnation threshold (consecutive iterations without improvement)"""

# Improvement detection update strategy
class ImprovementUpdateStrategy(UpdateStrategy):
    def __init__(self, threshold: int = 3):
        """threshold: Improvement threshold (consecutive improvements)"""

# Hybrid update strategy
class HybridUpdateStrategy(UpdateStrategy):
    def __init__(
        self,
        period: int = 10,
        stagnation_threshold: Optional[int] = None,
        improvement_threshold: Optional[int] = None
    ):
        """Combines periodic, stagnation, and improvement detection"""

# Composite update strategy
class CompositeUpdateStrategy(UpdateStrategy):
    def __init__(self, *strategies: UpdateStrategy):
        """Freely combine multiple strategies"""
```

## Advanced Usage

### Custom Compression Step

```python
from dimensio import CompressionStep
from ConfigSpace import ConfigurationSpace

class MyCustomStep(CompressionStep):
    def __init__(self, my_param):
        super().__init__(name='CustomStep', method='custom')
        self.my_param = my_param
    
    def compress(
        self,
        config_space: ConfigurationSpace,
        space_history=None,
        source_similarities=None
    ) -> ConfigurationSpace:
        # Implement your compression logic
        compressed_space = # ... your processing
        return compressed_space
    
    def affects_sampling_space(self) -> bool:
        return True  # Whether affects sampling space
    
    def needs_unproject(self) -> bool:
        return False  # Whether needs unprojection

# Use custom step
steps = [
    MyCustomStep(my_param=42),
    BoundaryRangeStep(method='boundary', top_ratio=0.8)
]
compressor = Compressor(config_space=config_space, steps=steps)
```

### Combining Multiple Strategies

```python
# Example 1: Dimension selection + Range compression + Projection
steps = [
    SHAPDimensionStep(strategy='shap', topk=20),        # Select 20 important params
    BoundaryRangeStep(method='boundary', top_ratio=0.8), # Compress to top-80% range
    REMBOProjectionStep(method='rembo', low_dim=10)     # Project to 10 dims
]

# Example 2: Quantization + Projection only (LlamaTune style)
steps = [
    QuantizationProjectionStep(method='quantization', max_num_values=50),
    HesBOProjectionStep(method='hesbo', low_dim=15)
]

# Example 3: Expert knowledge + Adaptive range compression
steps = [
    ExpertDimensionStep(strategy='expert', expert_params=['x1', 'x2', 'x3']),
    SHAPBoundaryRangeStep(method='shap_boundary', top_ratio=0.9)
]
```

### Handling Multi-source Transfer Learning Data

```python
from openbox.utils.history import History

# 1. Load multiple source task histories
source_histories = [history1, history2, history3]

# 2. Calculate task similarities (optional, for weighting)
source_similarities = {
    0: 0.8,  # Similarity of source task 0
    1: 0.6,
    2: 0.4
}

# 3. Compress with multi-source data
surrogate_space, sample_space = compressor.compress_space(
    space_history=source_histories,
    source_similarities=source_similarities
)

# 4. Transform source data to compressed space
transformed_histories = compressor.transform_source_data(source_histories)
```

### Dynamic Compression Strategy Update

Integration in BO:
```python
def _get_surrogate_config_array(self):
    X_surrogate = []
    for obs in self.history.observations:
        surrogate_config = self.compressor. \
            convert_config_to_surrogate_space(obs.config)
        X_surrogate.append(surrogate_config.get_array())
    return np.array(X_surrogate)

def update_compression(self, history):
    updated = self.compressor.update_compression(history)
    if updated:
        # compressor.update_compression already updated the spaces        
        # Rebuild surrogate model with new space dimensions

        self.surrogate = build_my_surrogate(
            # use surrogate_space here
            config_space=self.compressor.surrogate_space,
            # transform_source_data
            transfer_learning_history= \
                self.compressor.transform_source_data(
                    self.source_hpo_data
                ),
            ...
        )
        
        self.acq_optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acq_func,
            # use sample_space here
            config_space=self.compressor.sample_space,
            ...
        )
        
        X_surrogate = self._get_surrogate_config_array()
        Y = self.history.get_objectives()
        self.surrogate.train(X_surrogate, Y)
        
        self.acq_func.update(
            model=self.surrogate,
            eta=self.history.get_incumbent_value(),
            num_data=len(self.history)
        )
        return True 
    return False
```

```python
from dimensio import AdaptiveDimensionStep, Compressor
from dimensio.steps.dimension import SHAPImportanceCalculator
from dimensio.core.update import PeriodicUpdateStrategy

# Create adaptive dimension selection step
step = AdaptiveDimensionStep(
    importance_calculator=SHAPImportanceCalculator(),
    update_strategy=PeriodicUpdateStrategy(period=10),  # Check every 10 iterations
    initial_topk=30,
    reduction_ratio=0.2,  # Reduce by 20% each time
    min_dimensions=5,
    max_dimensions=50
)
compressor = Compressor(config_space=config_space, steps=[step])
# Then mount to advisor

# Auto-update in optimization loop
for iteration in range(max_iterations):
    # ... optimization logic
    
    # Periodically check if update needed
    updated = self.advisor.update_compression(history)

    if updated:
        print(f"Compression strategy updated (iteration {iteration})")
        # Update sampling strategy
        sampling_strategy = compressor.get_sampling_strategy()
```

### Save and Analyze Compression Info

```python
# 1. Enable compression info saving
compressor = Compressor(
    config_space=config_space,
    steps=steps,
    save_compression_info=True,
    output_dir='./results/compression'
)

# 2. Perform compression
compressor.compress_space()

# 3. Get compression summary
summary = compressor.get_compression_summary()
print(f"Original dimensions: {summary['original_dimensions']}")
print(f"Compressed dimensions: {summary['surrogate_dimensions']}")
print(f"Compression ratio: {summary['surrogate_compression_ratio']:.2%}")

# 4. View saved detailed info
# ./results/compression/compression_initial_compression_*.json
# ./results/compression/compression_history.json

# 5. Visualize
from dimensio.viz import visualize_compression_details
visualize_compression_details(compressor, save_dir='./results/viz')
```

## Dependencies

- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- ConfigSpace >= 0.6.0
- shap >= 0.41.0
- openbox >= 0.8.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Issues and Pull Requests are welcome!

## Author

Lingching Tung - lingchingtung@stu.pku.edu.cn

## Changelog

### 0.2.0 (2025-11-15)

#### Added
- Enhanced compression visualization coverage
- Added visualization tracking functionality
- Added documentation (README.md)
- Added example code directory (examples/)
  - Quick start example
  - Adaptive strategy example
  - Multi-source/single-source data example
  - Comprehensive examples

#### Fixed
- Fixed bug with duplicate names in utility module (logger => _logger)

### 0.1.0 (2025-11-13)

#### Added
- 🎉 Initial release of Dimensio
- Core compressor class `Compressor`
- Compression pipeline `CompressionPipeline`
- Three major compression strategy categories
- Flexible sampling strategies
- Filling strategies
- Standard logging system (based on Python logging)
- Convenience function `get_compressor()`
- Optimization progress tracking
- Multiple update strategies (periodic, stagnation detection, improvement detection, etc.)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{dimensio2025,
  author = {Lingching Tung},
  title = {Dimensio: Configuration Space Compression for Bayesian Optimization},
  year = {2025},
  url = {https://github.com/Elubrazione/dimensio}
}
```
