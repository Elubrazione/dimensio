# Dimensio

[English](./README.md) | 简体中文

一个灵活的配置空间压缩库，专为贝叶斯优化（Bayesian Optimization）设计。通过 Pipeline 架构支持多种压缩策略的组合，提升高维超参数优化的效率。

[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 目录

- [特性](#特性)
- [概览](#概览)
- [安装](#安装)
- [快速开始](#快速开始)
- [示例](#示例)
- [压缩策略](#压缩策略)
- [可视化功能](#可视化功能)
- [集成到贝叶斯优化系统](#集成到贝叶斯优化系统)
- [API 文档](#api-文档)
- [进阶用法](#进阶用法)

## 特性

✨ **Pipeline 架构**：通过组合多个压缩步骤构建灵活的压缩策略  
🎯 **多种压缩策略**：支持维度选择、范围压缩和投影变换  
🔄 **自适应更新**：根据优化过程动态调整压缩策略  
🎨 **丰富的可视化**：提供压缩过程、参数重要性等多种可视化工具  
📊 **迁移学习支持**：动态转换多源历史数据，适配压缩策略变化  
🔧 **可扩展设计**：易于添加自定义压缩步骤和填充策略

## 概览

### 空间概念

**原始空间（Original Space）**
- 完整的、未压缩的配置空间
- 包含所有参数及其原始范围

**采样空间（Sample Space）**
- 用于采样新配置的空间
- 受参数选择和范围裁剪影响
- 如果使用投影步骤，则为低维空间

**代理模型空间（Surrogate Space）**
- 代理模型训练和预测使用的空间

**去投影空间（Unprojected Space）**
- 投影前的空间
- 用于将低维配置映射回高维空间进行**评估**时的空间
  - 如果在投影步骤前进行了维度压缩或范围压缩，则此空间即为维度/范围压缩后、投影前的空间；若无，则为原空间。


### 压缩流程

```
原始空间（Original Space）
    ↓ [参数选择 - DimensionSelectionStep]
维度降低空间（Dimension-reduced Space）
    ↓ [范围裁剪 - RangeCompressionStep]
范围压缩空间（Range-compressed Space）
    ↓ [投影变换 - ProjectionStep]
最终返回压缩空间（Final Compressed Space）
    ├── 采样空间（Sample Space）：用于生成新配置
    └── 代理模型空间（Surrogate Space）：用于模型训练
```

## 安装

### 从 PyPI 安装

```bash
pip install dimensio
```

### 从源码安装

```bash
git clone https://github.com/Elubrazione/dimensio.git
cd dimensio
pip install -e .
```

## 快速开始

> 💡 **查看完整示例**：[examples/](./examples/) 目录包含多个可运行的完整示例，涵盖所有功能和使用场景。详见 [examples/README_CN.md](./examples/README_CN.md)。

### 基础使用
**强烈推荐自行组合压缩步骤 `step`**

```python
from dimensio import Compressor, SHAPDimensionStep, BoundaryRangeStep
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

# 1. 创建配置空间
config_space = ConfigurationSpace()
config_space.add_hyperparameter(UniformFloatHyperparameter('x1', 1, 100))
config_space.add_hyperparameter(UniformFloatHyperparameter('x2', -5, 1028))
config_space.add_hyperparameter(UniformFloatHyperparameter('x3', 3140, 7890))

# 2. 定义压缩步骤
steps = [
    SHAPDimensionStep(strategy='shap', topk=2),
    BoundaryRangeStep(method='boundary', top_ratio=0.8)
]

# 3. 创建压缩器
compressor = Compressor(
    config_space=config_space,
    steps=steps,
    save_compression_info=True,  # 保存压缩信息
    output_dir='./results/compression'
)

# 4. 压缩配置空间
surrogate_space, sample_space = compressor.compress_space(space_history=None)

print(f"原始维度: {len(config_space.get_hyperparameters())}")
print(f"代理模型空间维度: {len(surrogate_space.get_hyperparameters())}")
print(f"采样空间维度: {len(sample_space.get_hyperparameters())}")
```

### 使用便捷函数

```python
from dimensio import get_compressor

# LlamaTune 策略（量化 + 投影）
compressor = get_compressor(
    compressor_type='llamatune',
    config_space=config_space,
    adapter_alias='rembo',  # 或 'hesbo'
    le_low_dim=10,
    max_num_values=50
)

# 专家知识策略
compressor = get_compressor(
    compressor_type='expert',
    config_space=config_space,
    expert_params=['x1', 'x3'],
    top_ratio=0.9
)
```

### 配置日志

```python
from dimensio import setup_logging, disable_logging
import logging

# 设置日志级别
setup_logging(level=logging.INFO)

# 或保存到文件
setup_logging(level=logging.DEBUG, log_file='dimensio.log')

# 关闭日志
disable_logging()
```

## 示例

`examples/` 目录包含完整的可运行示例

### 1. 快速开始 (`quick_start.py`)
一个简单的示例，演示基本用法：
- 创建配置空间
- 生成模拟历史数据
- 使用便捷函数和自定义步骤
- 基本可视化

**运行**: `python examples/quick_start.py`

### 2. 综合示例 (`comprehensive.py`)
六个完整示例，涵盖不同的压缩策略：
- **示例 1**: SHAP 维度选择 + 普通边界范围压缩
- **示例 2**: 相关性维度选择 + SHAP 范围压缩
- **示例 3**: KDE 范围压缩（保留所有维度）
- **示例 4**: 量化 + REMBO 投影
- **示例 5**: 专家知识压缩
- **示例 6**: 使用便捷函数

**运行**: `python examples/comprehensive.py`

### 3. 自适应更新策略 (`adaptive_strategies.py`)
对比四种不同的自适应更新策略：
- **周期性更新**: 固定间隔更新
- **停滞检测**: 优化停滞时触发
- **改进检测**: 连续改进时触发
- **复合策略**: 组合多种策略（展示的是停滞检测+改进检测）

**运行**: `python examples/adaptive_strategies.py`

### 4. 多源任务迁移学习 (`multi_single_source.py`)
演示多源任务的迁移学习：
- 从多个源任务生成历史数据
- 计算任务相似度
- 对比单源与多源压缩效果
- 可视化迁移学习效果

**运行**: `python examples/multi_single_source.py`

所有示例的详细文档请参阅 [examples/README_CN.md](./examples/README_CN.md)。

## 压缩策略

### 1. 维度选择（Dimension Selection）

减少参数数量，保留最重要的参数。

#### SHAPDimensionStep

基于 SHAP 值的参数重要性选择。**支持多源迁移学习数据**。

```python
from dimensio import SHAPDimensionStep

step = SHAPDimensionStep(
    strategy='shap',
    topk=10  # 选择 top-10 重要参数
)
```

**工作原理**：
1. 使用历史评估数据训练随机森林回归模型
2. 计算 SHAP 值来量化每个参数的重要性
3. 选择重要性最高的 top-k 个参数

**迁移学习支持**：
- 可以利用多个源任务的历史数据
- 自动根据任务相似度加权不同源的重要性

#### CorrelationDimensionStep

基于 Spearman 或 Pearson 相关系数的参数选择。**支持多源迁移学习数据**。

```python
from dimensio import CorrelationDimensionStep

step = CorrelationDimensionStep(
    method='spearman',  # 或 'pearson'
    topk=10
)
```

**工作原理**：
- 计算每个参数与目标函数的相关系数（Spearman 或 Pearson）
- 选择相关性最高的参数

**迁移学习支持**：
- 可以利用多个源任务的历史数据
- 自动根据任务相似度加权不同源的重要性

#### ExpertDimensionStep

基于专家知识的参数选择。

```python
from dimensio import ExpertDimensionStep

step = ExpertDimensionStep(
    strategy='expert',
    expert_params=['param1', 'param2', 'param3']
)
```

#### AdaptiveDimensionStep

自适应调整参数数量。可配置重要性计算器和更新策略。

```python
from dimensio import AdaptiveDimensionStep
from dimensio.steps.dimension import SHAPImportanceCalculator
from dimensio.core.update import PeriodicUpdateStrategy

step = AdaptiveDimensionStep(
    importance_calculator=SHAPImportanceCalculator(),  # 可选，默认为 SHAP
    update_strategy=PeriodicUpdateStrategy(period=5),  # 每 5 次迭代更新
    initial_topk=30,
    reduction_ratio=0.2,
    min_dimensions=5,
    max_dimensions=50  # 可选
)
```

**参数说明**：
- `importance_calculator`: 重要性计算器（默认 SHAP）
- `update_strategy`: 更新策略（默认每 5 次迭代），详见下方说明
- `initial_topk`: 初始参数数量
- `reduction_ratio`: 每次调整的比例（用于增加或减少维度）
- `min_dimensions`: 最小维度数
- `max_dimensions`: 最大维度数（可选）

**支持的更新策略**：

##### 1. PeriodicUpdateStrategy（周期性更新）

每隔固定迭代次数执行一次更新，逐步减少参数数量。

```python
from dimensio.core.update import PeriodicUpdateStrategy

update_strategy = PeriodicUpdateStrategy(period=10)  # 每 10 次迭代更新一次
```

**行为**：每 `period` 次迭代后，减少 `current_topk × reduction_ratio` 个参数。

##### 2. StagnationUpdateStrategy（停滞检测更新）

检测到优化停滞时，增加参数数量以扩大搜索空间。

```python
from dimensio.core.update import StagnationUpdateStrategy

update_strategy = StagnationUpdateStrategy(threshold=5)  # 停滞 5 次迭代后触发
```

**行为**：当最优值连续 `threshold` 次迭代未改善时，增加 `current_topk × reduction_ratio` 个参数。

##### 3. ImprovementUpdateStrategy（改进检测更新）

检测到优化有改进时，减少参数数量以聚焦搜索。

```python
from dimensio.core.update import ImprovementUpdateStrategy

update_strategy = ImprovementUpdateStrategy(threshold=3)  # 连续改进 3 次触发
```

**行为**：当最优值连续 `threshold` 次迭代都有改善时，减少 `current_topk × reduction_ratio` 个参数。

##### 4. HybridUpdateStrategy（混合更新策略）

结合周期性、停滞检测和改进检测的混合策略。

```python
from dimensio.core.update import HybridUpdateStrategy

update_strategy = HybridUpdateStrategy(
    period=10,                    # 基础周期：每 10 次迭代
    stagnation_threshold=5,       # 停滞检测：5 次未改进
    improvement_threshold=3       # 改进检测：连续 3 次改进
)
```

**行为**：
- **优先级**：停滞检测 > 改进检测 > 周期性
- 停滞时：增加维度
- 改进时：减少维度
- 周期到达时：减少维度

##### 5. CompositeUpdateStrategy（组合更新策略）

自由组合多个策略，任一策略触发即执行更新。

```python
from dimensio.core.update import CompositeUpdateStrategy, StagnationUpdateStrategy, ImprovementUpdateStrategy

update_strategy = CompositeUpdateStrategy(
    StagnationUpdateStrategy(threshold=5),
    ImprovementUpdateStrategy(threshold=3)
)
```

**行为**：按顺序检查每个策略，第一个触发的策略决定如何更新维度。

**使用建议**：
- **稳定优化**：使用 `PeriodicUpdateStrategy`，平稳减少维度
- **容易停滞**：使用 `StagnationUpdateStrategy` 或 `HybridUpdateStrategy`
- **快速收敛**：使用 `ImprovementUpdateStrategy`
- **复杂场景**：使用 `HybridUpdateStrategy` 或 `CompositeUpdateStrategy`

### 2. 范围压缩（Range Compression）

缩小参数值域，聚焦到高价值区域。

#### BoundaryRangeStep

基于历史最优配置的均值和标准差压缩范围。

```python
from dimensio import BoundaryRangeStep

step = BoundaryRangeStep(
    method='boundary',
    top_ratio=0.8,  # 使用 top-80% 的配置计算边界
    sigma=2.0       # 标准差倍数（μ ± 2σ）
)
```

#### SHAPBoundaryRangeStep

基于 SHAP 值加权的范围压缩。**支持多源迁移学习数据**。

```python
from dimensio import SHAPBoundaryRangeStep

step = SHAPBoundaryRangeStep(
    method='shap_boundary',
    top_ratio=0.8,
    sigma=2.0
)
```

**工作原理**：
- 根据参数重要性调整压缩程度
- 重要参数保留更大的搜索范围

#### KDEBoundaryRangeStep

基于核密度估计的范围压缩。**支持多源迁移学习数据**。

```python
from dimensio import KDEBoundaryRangeStep

step = KDEBoundaryRangeStep(
    method='kde_boundary',
    source_top_ratio=0.3,  # 使用源任务 top-30% 的配置
    kde_coverage=0.6       # KDE 覆盖率（保留概率密度的 60%）
)
```

**工作原理**：
- 使用 KDE (Kernel Density Estimation) 估计参数的概率密度分布
- 根据 `kde_coverage` 确定保留高密度区域的比例
- 对多源任务数据，根据任务相似度加权处理

#### ExpertRangeStep

基于专家指定的参数范围。

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

### 3. 投影变换（Projection）

变换参数表示方式，降低搜索复杂度。

#### QuantizationProjectionStep

整数参数量化，将大范围整数参数压缩到更小的离散值集合。

```python
from dimensio import QuantizationProjectionStep

step = QuantizationProjectionStep(
    method='quantization',
    max_num_values=50,  # 最大离散值数量
    adaptive=False  # 是否自适应调整
)
```

**工作原理**：
- 仅对 `UniformIntegerHyperparameter` 类型且值域大于 `max_num_values` 的参数进行量化
- 将原始范围 `[lower, upper]` 映射到量化范围 `[1, max_num_values]`
- 采样时在量化空间生成整数，评估时反投影回原始范围
- 其他类型参数保持不变

**示例**：
- 原始参数：`x ∈ [1000, 5000]` (4001 个值)
- 量化后：`x|q ∈ [1, 50]` (50 个值)
- 压缩比：50/4001 ≈ 1.25%

#### REMBOProjectionStep

随机嵌入贝叶斯优化（Random Embedding Bayesian Optimization）。

```python
from dimensio import REMBOProjectionStep

step = REMBOProjectionStep(
    method='rembo',
    low_dim=10,  # 低维空间维度
    max_num_values=50  # 配合量化使用
)
```

**工作原理**：
- 假设高维空间存在低维有效子空间
- 通过随机矩阵将 d 维投影到 d_e 维（d_e << d）
- 低维采样范围：`[-√d_e, √d_e]`

#### HesBOProjectionStep

哈希嵌入贝叶斯优化（Hashing Embedding Bayesian Optimization）。

```python
from dimensio import HesBOProjectionStep

step = HesBOProjectionStep(
    method='hesbo',
    low_dim=10,
    max_num_values=50
)
```

**工作原理**：
- 使用哈希函数进行维度映射
- 低维采样范围：`[-1, 1]`
- 比 REMBO 更节省内存

#### KPCAProjectionStep

核主成分分析投影。

```python
from dimensio import KPCAProjectionStep

step = KPCAProjectionStep(
    method='kpca',
    n_components=10,
    kernel='rbf'
)
```

**工作原理**：
- 使用核方法提取非线性主成分
- **注意事项：** 该方法只是使用提取后的主成分维度训练代理模型，返回的采样空间仍为该步骤前的空间

## 可视化功能

Dimensio 提供丰富的可视化工具来分析压缩效果。可视化系统会**自动检测**使用的压缩步骤并生成相关图表。

### 自动可视化

```python
from dimensio.viz import visualize_compression_details

# 根据使用的步骤自动生成所有相关可视化
visualize_compression_details(
    compressor=compressor,
    save_dir='./results/visualization'
)
```

**生成的图表**（根据压缩步骤自动选择）：

1. **compression_summary.png**：压缩总结
   - 各步骤维度变化
   - 压缩比率统计
   - 范围压缩统计
   - 文字摘要

2. **range_compression_step_*.png**：每个范围压缩步骤的详细视图
   - 自动检测：使用了 `BoundaryRangeStep`、`SHAPBoundaryRangeStep`、`KDEBoundaryRangeStep` 等
   - 原始范围 vs 压缩范围
   - 每个参数的压缩比率
   - 量化信息（如果使用）

3. **parameter_importance_step_*.png**：参数重要性可视化
   - 自动检测：使用了 `SHAPDimensionStep`、`CorrelationDimensionStep`、`AdaptiveDimensionStep`
   - Top-K 参数的重要性分数

4. **dimension_evolution.png**：维度演化曲线
   - 自动检测：使用了 `AdaptiveDimensionStep` 且有更新历史
   - 显示迭代过程中维度数量的变化
   - 高亮标注每次维度调整

5. **source_task_similarities.png**：源任务相似度
   - 自动检测：使用了多源任务（≥2个任务）迁移学习（提供了 `source_similarities`）
   - 各源任务与目标任务的相似度分数柱状图

6. **multi_task_importance_heatmap_step_*.png**：多任务参数重要性热力图
   - 自动检测：使用了 SHAP/Correlation-based 的维度压缩方法 + 多个源任务
   - 不同任务对各个参数的重要性对比热力图
   - 用途：发现跨任务的通用重要参数和任务特定的关键参数


## 集成到贝叶斯优化系统
Dimensio 可以无缝集成到贝叶斯优化系统中
- 使用compressor的转换接口 `transform_source_data` 转换自动历史数据
- 使用 `surrogate_space` 训练代理模型，使用 `sample_space` 进行数据采样
- 若采样的为投影后的配置可通过 `compressor.unproject_point()` 进行转换，转换后的空间可通过 `compressor.get_unprojected_space()` 获取


## API 文档

### Compressor

主压缩器类，管理压缩 pipeline 和配置空间转换。

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
        参数：
            config_space: 原始配置空间
            steps: 压缩步骤列表
            filling_strategy: 填充策略（默认使用搜索空间默认值填充）
            save_compression_info: 是否保存压缩信息
            output_dir: 输出目录
        """
```

**主要方法**：

```python
def compress_space(
    self,
    space_history: Optional[List] = None,
    source_similarities: Optional[Dict[int, float]] = None
) -> Tuple[ConfigurationSpace, ConfigurationSpace]:
    """
    压缩配置空间
    
    参数：
        space_history: 历史数据（用于SHAP、KDE等方法）
        source_similarities: 源任务相似度（用于迁移学习）
    
    返回：
        (surrogate_space, sample_space)
    """

def transform_source_data(
    self,
    source_hpo_data: Optional[List[History]]
) -> Optional[List[History]]:
    """转换源任务数据到当前压缩空间"""

def convert_config_to_surrogate_space(
    self,
    config: Configuration
) -> Configuration:
    """将配置转换到代理模型空间"""

def get_compression_summary(self) -> dict:
    """获取压缩摘要信息"""

def unproject_point(self, point: Configuration) -> dict:
    """反投影配置（投影步骤 -> 原始空间）"""

def update_compression(self, history: History) -> bool:
    """自适应更新压缩策略"""

def get_sampling_strategy(self) -> SamplingStrategy:
    """获取采样策略"""
```

### FillingStrategy

填充策略接口，用于处理维度变化时的参数填充。

```python
class FillingStrategy(ABC):
    @abstractmethod
    def fill_missing_parameters(
        self,
        config_dict: Dict[str, Any],
        target_space: ConfigurationSpace
    ) -> Dict[str, Any]:
        """填充缺失的参数"""

# 默认填充策略（使用搜索空间默认值）
class DefaultValueFilling(FillingStrategy):
    ...

# 裁剪填充策略（裁剪到范围内）
class ClippingValueFilling(FillingStrategy):
    ...
```

## 进阶用法

### 自定义压缩步骤

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
        # 实现你的压缩逻辑
        compressed_space = # ... 你的处理
        return compressed_space
    
    def affects_sampling_space(self) -> bool:
        return True  # 是否影响采样空间
    
    def needs_unproject(self) -> bool:
        return False  # 是否需要反投影

# 使用自定义步骤
steps = [
    MyCustomStep(my_param=42),
    BoundaryRangeStep(method='boundary', top_ratio=0.8)
]
compressor = Compressor(config_space=config_space, steps=steps)
```

### 组合多种压缩策略

```python
# 示例 1：参数选择 + 范围压缩 + 投影
steps = [
    SHAPDimensionStep(strategy='shap', topk=20),        # 选 20 个重要参数
    BoundaryRangeStep(method='boundary', top_ratio=0.8), # 压缩到 top-80% 范围
    REMBOProjectionStep(method='rembo', low_dim=10)     # 投影到 10 维
]

# 示例 2：仅量化 + 投影（LlamaTune 风格）
steps = [
    QuantizationProjectionStep(method='quantization', max_num_values=50),
    HesBOProjectionStep(method='hesbo', low_dim=15)
]

# 示例 3：专家知识 + 自适应范围压缩
steps = [
    ExpertDimensionStep(strategy='expert', expert_params=['x1', 'x2', 'x3']),
    SHAPBoundaryRangeStep(method='shap_boundary', top_ratio=0.9)
]
```

### 处理多源迁移学习数据

```python
from openbox.utils.history import History

# 1. 加载多个源任务的历史数据
source_histories = [history1, history2, history3]

# 2. 计算任务相似度（可选，用于加权）
source_similarities = {
    0: 0.8,  # 源任务 0 的相似度
    1: 0.6,
    2: 0.4
}

# 3. 使用多源数据压缩
surrogate_space, sample_space = compressor.compress_space(
    space_history=source_histories,
    source_similarities=source_similarities
)

# 4. 转换源数据到压缩空间
transformed_histories = compressor.transform_source_data(source_histories)
```

### 动态压缩策略更新

在BO中集成
```py
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

# 创建自适应维度选择步骤
step = AdaptiveDimensionStep(
    importance_calculator=SHAPImportanceCalculator(),
    update_strategy=PeriodicUpdateStrategy(period=10),  # 每 10 次迭代检查一次
    initial_topk=30,
    reduction_ratio=0.2,  # 每次减少 20%
    min_dimensions=5,
    max_dimensions=50
)
compressor = Compressor(config_space=config_space, steps=[step])
# 然后挂载到 advisor 中

# 在优化循环中自动更新
for iteration in range(max_iterations):
    # ... 优化逻辑
    
    # 定期检查是否需要更新
    updated = self.advisor.update_compression(history)

    if updated:
        print(f"压缩策略已更新（第 {iteration} 次迭代）")
        # 更新采样策略
        sampling_strategy = compressor.get_sampling_strategy()
```

### 保存和分析压缩信息

```python
# 1. 启用压缩信息保存
compressor = Compressor(
    config_space=config_space,
    steps=steps,
    save_compression_info=True,
    output_dir='./results/compression'
)

# 2. 执行压缩
compressor.compress_space()

# 3. 获取压缩摘要
summary = compressor.get_compression_summary()
print(f"原始维度: {summary['original_dimensions']}")
print(f"压缩后维度: {summary['surrogate_dimensions']}")
print(f"压缩比: {summary['surrogate_compression_ratio']:.2%}")

# 4. 查看保存的详细信息
# ./results/compression/compression_initial_compression_*.json
# ./results/compression/compression_history.json

# 5. 可视化
from dimensio.viz import visualize_compression_details
visualize_compression_details(compressor, save_dir='./results/viz')
```

## 依赖项

- numpy >= 1.19.0
- pandas >= 1.2.0
- scikit-learn >= 0.24.0
- ConfigSpace >= 0.6.0
- shap >= 0.41.0
- openbox >= 0.8.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 Pull Request！

## 作者

Lingching Tung - lingchingtung@stu.pku.edu.cn

## 更新日志

### 0.2.0 (2025-11-15)

#### 新增
- 增强压缩可视化覆盖范围
- 添加可视化跟踪功能
- 添加中文文档（README_CN.md）
- 添加示例代码目录（examples/）
  - 快速开始示例
  - 自适应策略示例
  - 多源/单源数据示例
  - 综合示例

#### 修复
- 修复工具模块中重复名称的 bug（logger => _logger）

### 0.1.0 (2025-11-13)

#### 新增
- 🎉 Dimensio 初始版本发布
- 核心压缩器类 `Compressor`
- 压缩管道 `CompressionPipeline`
- 三大类压缩策略
- 灵活的采样策略
- 填充策略
- 标准日志系统（基于 Python logging）
- 便捷函数 `get_compressor()`
- 优化进度跟踪
- 多种更新策略（周期性、停滞检测、改进检测等）

## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{dimensio2025,
  author = {Lingching Tung},
  title = {Dimensio: Configuration Space Compression for Bayesian Optimization},
  year = {2025},
  url = {https://github.com/Elubrazione/dimensio}
}
```
