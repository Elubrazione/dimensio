# Frontend API 使用指南

本文档介绍如何从前端调用 Dimensio 压缩服务


## 快速开始

### 命令行调用方式

**分别创建配置文件：**

1. **配置空间文件** `config_space.json`：
```json
{
  "spark.memory.fraction": {
    "type": "float",
    "min": 0.1,
    "max": 0.9,
    "default": 0.6
  },
  "spark.stage.maxConsecutiveAttempts": {
    "type": "integer",
    "min": 1,
    "max": 10,
    "default": 4
  }
}
```

2. **步骤配置文件** `steps.json`：
```json
{
  "dimension_step": "d_shap",
  "range_step": "r_boundary",
  "projection_step": "p_none",
  "step_params": {
    "d_shap": {"topk": 3}
  }
}
```

3. **历史数据文件** `history.json`（必需，单个或多个）：
```json
[
  {
    "config": {
      "spark.memory.fraction": 0.6,
      "spark.stage.maxConsecutiveAttempts": 4
    },
    "objectives": [52.5],
    "constraints": null,
    "trial_state": 0
  }
]
```

**注意：** 
- 每个 JSON 文件包含一个观察列表（数组格式）
- 系统会将每个 JSON 文件转换为一个 `History` 对象
- 多个历史文件时，格式为 `[[...], [...], ...]`，每个内层列表对应一个 JSON 文件

**运行命令：**

```bash
# 单个历史文件
python -m dimensio.api.compress_api \
  --config-space config_space.json \
  --steps steps.json \
  --history history.json \
  --output-dir ./results

# 多个历史文件（多源迁移学习）
# 源相似度会自动计算为 1/len(histories) 每个历史文件
python -m dimensio.api.compress_api \
  --config-space config_space.json \
  --steps steps.json \
  --history history1.json history2.json history3.json \
  --output-dir ./results
```

## 配置格式说明

### 配置空间定义 (config_space)

每个超参数的定义格式：

```json
{
  "param_name": {
    "type": "float" | "integer" | "int" | "categorical",
    "min": number,           // float/integer/int 必需
    "max": number,           // float/integer/int 必需
    "default": value,        // 可选
    "log": boolean,          // 可选，仅 float/integer/int
    "choices": [array]       // categorical 必需
  }
}
```

**注意：** `"type"` 字段支持 `"integer"` 和 `"int"` 两种格式（不区分大小写），推荐使用 `"integer"`。

**示例：**

- **Float 参数：**
```json
{
  "spark.memory.fraction": {
    "type": "float",
    "min": 0.1,
    "max": 0.9,
    "default": 0.6
  }
}
```

- **Integer 参数（推荐格式）：**
```json
{
  "spark.stage.maxConsecutiveAttempts": {
    "type": "integer",
    "min": 1,
    "max": 10,
    "default": 4
  }
}
```

- **Integer 参数（兼容格式）：**
```json
{
  "batch_size": {
    "type": "int",
    "min": 16,
    "max": 512,
    "default": 128,
    "log": true
  }
}
```

- **Categorical 参数：**
```json
{
  "optimizer": {
    "type": "categorical",
    "choices": ["adam", "sgd", "rmsprop"],
    "default": "adam"
  }
}
```

### 步骤配置 (steps)

```json
{
  "dimension_step": "d_shap" | "d_corr" | "d_expert" | "d_adaptive" | "d_none",
  "range_step": "r_boundary" | "r_shap" | "r_kde" | "r_expert" | "r_none",
  "projection_step": "p_quant" | "p_rembo" | "p_hesbo" | "p_kpca" | "p_none",
  "step_params": {
    "d_shap": {"topk": 10},
    "r_kde": {"source_top_ratio": 0.5, "kde_coverage": 0.7}
  }
}
```

**可用的步骤字符串：**

- **Dimension Steps:**
  - `d_shap` - SHAP 维度选择
  - `d_corr` - 相关性维度选择
  - `d_expert` - 专家知识维度选择
  - `d_adaptive` - 自适应维度选择
  - `d_none` - 无维度选择

- **Range Steps:**
  - `r_boundary` - 边界范围压缩
  - `r_shap` - SHAP 加权边界范围压缩
  - `r_kde` - KDE 边界范围压缩
  - `r_expert` - 专家指定范围压缩
  - `r_none` - 无范围压缩

- **Projection Steps:**
  - `p_quant` - 量化投影
  - `p_rembo` - REMBO 投影
  - `p_hesbo` - HesBO 投影
  - `p_kpca` - 核 PCA 投影
  - `p_none` - 无投影

**步骤参数覆盖 (step_params):**

可以通过 `step_params` 覆盖任何步骤的默认参数：

```json
{
  "step_params": {
    "d_shap": {
      "topk": 5  // 覆盖默认值 20
    },
    "r_kde": {
      "source_top_ratio": 0.5,  // 覆盖默认值 0.3
      "kde_coverage": 0.8        // 覆盖默认值 0.6
    },
    "p_quant": {
      "max_num_values": 20,     // 覆盖默认值 10
      "adaptive": true           // 覆盖默认值 false
    }
  }
}
```

### 历史数据 (history)

**必需**，用于提供历史评估数据以指导压缩。**支持单个或多个历史文件**，适用于多源迁移学习场景。

**数据格式：**
- 输入格式：`List[List[Dict]]` - 二维数组，每个内层列表对应一个 JSON 文件
- 处理方式：每个 JSON 文件（内层列表）会被转换为一个 `History` 对象
- 最终格式：`List[History]` - 传递给压缩引擎

**支持两种 JSON 格式：**

**格式 1：简单格式（向后兼容）**
```json
[
  {
    "config": {
      "learning_rate": 0.001,
      "batch_size": 128
    },
    "objective": 10.5,
    "trial_state": "SUCCESS",
    "elapsed_time": 0.5
  }
]
```

**格式 2：完整格式（推荐，来自 History 对象）**
```json
[
  {
    "config": {
      "spark.executor.memory": 52,
      "spark.executor.cores": 16,
      "spark.memory.fraction": 0.6
    },
    "objectives": [52.489781712289336],
    "constraints": null,
    "trial_state": 0,
    "elapsed_time": 0.3046904864338654,
    "create_time": "2025-11-02T19:42:52.612589",
    "extra_info": {
      "origin": "Default Configuration"
    }
  }
]
```

**字段说明：**
- `config`: 必需，参数字典
- `objectives` 或 `objective`: 必需，目标值（`objectives` 为数组，`objective` 为单个值）
- `constraints`: 可选，约束值数组或 null
- `trial_state`: 可选，试验状态（`0` 表示 SUCCESS，或字符串 `"SUCCESS"`）
- `elapsed_time`: 可选，运行时间（秒）
- `create_time`: 可选，创建时间（会被忽略）
- `extra_info`: 可选，额外信息（会被忽略）

**多源迁移学习：**

当有多个源任务的历史数据时，可以上传多个历史文件。**源相似度会自动计算为 `1/len(histories)` 每个历史文件**，无需手动指定。

**处理流程：**
1. 每个 JSON 文件（如 `history1.json`）包含一个观察列表：`[{...}, {...}, ...]`
2. 系统将每个 JSON 文件转换为一个 `History` 对象
3. 多个文件形成 `List[History]` 格式：`[History1, History2, History3]`
4. 每个 `History` 对应一个源任务，相似度自动计算为 `1/3 = 0.333`

```bash
# 多个历史文件（多源迁移学习）
python -m dimensio.api.compress_api \
  --config-space config_space.json \
  --steps steps.json \
  --history history1.json history2.json history3.json
```

## 返回结果格式

```json
{
  "success": true,
  "original_dim": 4,
  "surrogate_dim": 2,
  "sample_dim": 2,
  "compression_ratio": 0.5,
  "original_params": ["param1", "param2", ...],
  "surrogate_params": ["param1", ...],
  "sample_params": ["param1", ...],
  "steps_used": ["SHAPDimensionStep", "BoundaryRangeStep"],
  "output_dir": "./results/compression",
  "compression_summary": {
    "original_dimensions": 4,
    "surrogate_dimensions": 2,
    "sample_dimensions": 2,
    "sample_compression_ratio": 0.5,
    "surrogate_compression_ratio": 0.5
  }
}
```

## 错误处理

如果发生错误，返回格式：

```json
{
  "success": false,
  "error": "Error message",
  "error_type": "ValueError"
}
```

## 命令行选项

```bash
python -m dimensio.api.compress_api [options]

必需选项:
  --config-space FILE    配置空间定义文件（JSON）
  --steps FILE           步骤配置文件（JSON）
  --history FILE [FILE ...]  历史数据文件（必需，可指定多个，用于多源迁移学习）
                             源相似度会自动计算为 1/len(histories) 每个历史文件

可选选项:
  --output-dir DIR           输出目录（默认: ./results/compression）
  --no-save                 不保存压缩信息
  --verbose                 启用详细日志
```

**示例：**

```bash
# 单个历史文件
python -m dimensio.api.compress_api \
  --config-space config_space.json \
  --steps steps.json \
  --history history.json

# 多个历史文件（多源迁移学习）
# 源相似度会自动计算为 1/3 = 0.333 每个历史文件
python -m dimensio.api.compress_api \
  --config-space config_space.json \
  --steps steps.json \
  --history source1.json source2.json source3.json \
  --output-dir ./results \
  --verbose
```