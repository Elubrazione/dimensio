# Dimensio 示例

本目录包含 Dimensio 的使用示例，展示各种压缩策略和可视化功能。

## 📦 工具模块

### `utils.py` - 通用工具函数

提供了常用的配置空间创建和数据生成函数：

**配置空间创建**:
- `create_simple_config_space(n_float, n_int)` - 创建简单的参数空间
- `create_spark_config_space()` - 创建 Spark 配置空间

**目标函数**:
- `simple_objective(config_dict)` - 简单的目标函数
- `spark_objective(config_dict)` - Spark 配置目标函数

**数据生成**:
- `generate_history(config_space, n_samples, ...)` - 生成模拟历史数据
- `generate_mock_history(config_space, n_samples, ...)` - 生成模拟历史数据（别名）
- `generate_improving_history(config_space, iteration, ...)` - 生成改进趋势的数据

**使用示例**:
```python
from examples.utils import create_simple_config_space, generate_history

config_space = create_simple_config_space(n_float=10, n_int=5)
history = generate_history(config_space, n_samples=50)
```

## 📚 示例列表

### 1. 快速开始 (`quick_start.py`)

**适合**: 初学者  
**内容**: 
- 创建简单的配置空间
- 生成模拟历史数据
- 使用便捷函数和自定义步骤
- 基本的可视化

**运行**:
```bash
cd examples
python quick_start.py
```

**输出**: `./results/quick_start/`

**关键代码片段**:
```python
from dimensio import get_compressor, SHAPDimensionStep, BoundaryRangeStep

# 方式1: 使用便捷函数
compressor = get_compressor(
    compressor_type='shap',
    config_space=config_space,
    topk=3,
    top_ratio=0.8
)

# 方式2: 自定义步骤组合
steps = [
    SHAPDimensionStep(strategy='shap', topk=4),
    BoundaryRangeStep(method='boundary', top_ratio=0.7, sigma=2.0)
]
compressor = Compressor(config_space=config_space, steps=steps)
```

---

### 2. 综合示例 (`comprehensive.py`)

**适合**: 了解所有功能的用户  
**内容**: 6 个完整示例，涵盖：

#### 示例 1: SHAP 维度选择 + 普通边界范围压缩
- **策略**: `SHAPDimensionStep` + `BoundaryRangeStep`
- **适用**: 数据驱动的参数选择和范围优化
- **输出**: `./results/comprehensive/example1_shap_boundary/`

#### 示例 2: 相关性维度选择 + SHAP 范围压缩
- **策略**: `CorrelationDimensionStep` + `SHAPBoundaryRangeStep`
- **适用**: 快速相关性分析 + 重要性加权范围压缩
- **输出**: `./results/comprehensive/example2_correlation_shap/`

#### 示例 3: KDE 范围压缩
- **策略**: `KDEBoundaryRangeStep`（仅范围压缩，保留所有维度）
- **适用**: 基于密度估计的范围优化
- **输出**: `./results/comprehensive/example3_kde/`

#### 示例 4: 量化 + REMBO 投影
- **策略**: `QuantizationProjectionStep` + `REMBOProjectionStep`
- **适用**: 高维空间降维，LlamaTune 风格
- **输出**: `./results/comprehensive/example4_quantization_rembo/`

#### 示例 5: 专家知识
- **策略**: `ExpertDimensionStep` + `ExpertRangeStep`
- **适用**: 基于领域知识的手动配置
- **输出**: `./results/comprehensive/example5_expert/`

#### 示例 6: 便捷函数
- **策略**: 使用 `get_compressor()` 快速创建
- **展示**: SHAP、LlamaTune、Expert 三种预设策略

**运行**:
```bash
cd examples
python comprehensive.py
```

**输出**: `./results/comprehensive/example{1-6}_*/`

---

### 3. 自适应更新策略对比 (`adaptive_strategies.py`)

**适合**: 想要深入了解自适应更新策略的用户  
**内容**: 对比4种不同的更新策略效果

**策略说明**:
- **周期性更新** (Periodic): 每3次迭代自动更新
- **停滞检测** (Stagnation): 连续3次无改进时触发
- **改进检测** (Improvement): 连续2次改进时触发  
- **复合策略** (Composite): **混合数据模式**
  - 迭代 0-4: 改进阶段 → 触发 Improvement
  - 迭代 5-9: 停滞阶段 → 触发 Stagnation
  - 迭代 10-14: 再次改进 → 触发 Improvement

#### 4种更新策略：

1. **Periodic Strategy (周期性策略)**
   - 每N次迭代自动触发更新
   - 适用场景：定期调整，不依赖性能变化
   - 参数：`period=3` (每3次迭代触发一次)

2. **Stagnation Detection (停滞检测)**
   - 当性能连续N次没有改进时触发
   - 适用场景：性能停滞时增加维度探索更多空间
   - 参数：`threshold=3` (连续3次停滞)
   - 行为：增加维度

3. **Improvement Detection (改进检测)**
   - 当性能连续N次持续改进时触发
   - 适用场景：性能良好时减少维度提高效率
   - 参数：`threshold=2` (连续2次改进)
   - 行为：减少维度

4. **Composite Strategy (组合策略)**
   - 组合多个策略，任一触发即更新
   - 适用场景：平衡探索与利用
   - 示例：`Stagnation + Improvement`

**运行**:
```bash
cd examples
python adaptive_strategies.py
```

**输出**: 
- `./results/adaptive_strategies/periodic/` - 周期性策略结果
- `./results/adaptive_strategies/stagnation/` - 停滞检测策略结果
- `./results/adaptive_strategies/improvement/` - 改进检测策略结果
- `./results/adaptive_strategies/composite/` - 复合策略结果
- `./results/adaptive_strategies/adaptive_strategies_comparison.png` - 四宫格对比图

**关键代码片段**:
```python
from dimensio import AdaptiveDimensionStep
from dimensio.core.update import (
    PeriodicUpdateStrategy,
    StagnationUpdateStrategy,
    ImprovementUpdateStrategy,
    CompositeUpdateStrategy
)

# 周期性策略
step = AdaptiveDimensionStep(
    update_strategy=PeriodicUpdateStrategy(period=3),
    initial_topk=10,
    reduction_ratio=0.2,
    min_dimensions=4,
    max_dimensions=12
)

# 复合策略
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

### 4. 多源任务迁移学习 (`multi_single_source.py`)

**适合**: 想要利用历史任务数据加速新任务优化的用户  
**内容**: 展示如何使用多个源任务的历史数据来优化新的目标任务

#### 场景说明：
模拟Spark不同工作负载类型的配置优化：
- **源任务 1**: Sort 工作负载（过往优化数据）
- **源任务 2**: Join 工作负载（过往优化数据）
- **源任务 3**: Aggregate 工作负载（过往优化数据）
- **目标任务**: Group-by 工作负载（新任务，需要优化）

#### 核心功能：
1. **任务相似度计算** - 基于领域知识或元特征计算任务间相似度
2. **加权迁移学习** - 根据相似度对源任务数据加权
3. **相似度可视化** - 自动生成任务相似度热力图


**运行**:
```bash
cd examples
python multi_single_source.py
```

**输出**: 
- `./results/multiple_single_source/multiple_source/` - 使用多源任务迁移学习的压缩结果
- `./results/multiple_single_source/single_source/` - 使用当前任务数据的结果
- 包含 `source_task_similarities.png` - 源任务相似度可视化
- 包含 `multi_task_importance_heatmap.png` - 多任务参数重要性热力图

**关键代码片段**:
```python
from dimensio import SHAPDimensionStep, SHAPBoundaryRangeStep

# 计算任务相似度
source_similarities = {
    0: 0.65,  # Sort 工作负载相似度
    1: 0.80,  # Join 工作负载相似度
    2: 0.75   # Aggregate 工作负载相似度
}

# 多源任务压缩
compressor = Compressor(
    config_space=config_space,
    steps=[
        SHAPDimensionStep(topk=6),
        SHAPBoundaryRangeStep(top_ratio=0.75)
    ]
)

# 使用多源历史数据压缩
surrogate_space, sample_space = compressor.compress_space(
    space_history=source_histories + [target_history],
    source_similarities=source_similarities
)
```

## 📊 输出文件

每个示例还会生成：

- **compression_initial_compression_*.json** - 初始压缩详情
- **compression_history.json** - 压缩历史记录（如果有更新）
- 包含所有步骤的详细信息和压缩统计


## 📖 更多信息

- [主文档（英文）](../README.md)
- [中文文档](../README_CN.md)
- [API 文档](../docs/)
- [发布指南](../docs/PUBLISH_GUIDE.md)

## 📝 贡献示例

如果您想贡献新的示例：

1. 在 `examples/` 目录下创建新的 Python 文件
2. 使用 `utils.py` 中的工具函数
3. 添加详细的文档字符串
4. 确保示例可以独立运行
5. 更新当前 README.md 文件

## 📧 反馈

如果您在使用示例时遇到问题或有改进建议，欢迎：
- 提交 Issue: [GitHub Issues](https://github.com/Elubrazione/dimensio/issues)
- 发送邮件: lingchingtung@stu.pku.edu.cn

