# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-11-13

### Added
- 初始版本发布
- 核心压缩器类 `Compressor`
- 压缩管道 `CompressionPipeline`
- 三大类压缩策略：
  - 维度选择（Dimension Selection）
    - SHAP-based 维度选择
    - 专家知识维度选择
    - 相关性维度选择
    - 自适应维度选择
  - 范围压缩（Range Compression）
    - 边界范围压缩
    - 专家范围压缩
    - SHAP 边界范围压缩
    - KDE 边界范围压缩
  - 投影变换（Projection）
    - REMBO 投影
    - HesBO 投影
    - KPCA 投影
    - 量化投影
- 灵活的采样策略
  - 标准采样
  - 混合范围采样
- 填充策略
  - 默认值填充
  - 裁剪填充
- 标准日志系统（基于 Python logging）
- 便捷函数 `get_compressor()`
- 优化进度跟踪
- 多种更新策略（周期性、停滞检测、改进检测等）

### Changed


### Fixed


## [0.2.2] - 2025-11-18

### Added
- 前端 API 支持：`compress_from_config` 函数，支持从 JSON 配置执行压缩
- 步骤工厂 API：`create_step_from_string` 和 `create_steps_from_strings`，支持从字符串标识符创建压缩步骤
- 命令行接口：`python -m dimensio.api.compress_api`，支持分开上传配置文件
- 前端集成文档：`examples/FRONTEND_API.md`


## [0.2.0] - 2025-11-15

### Added
- 增强压缩可视化覆盖范围
- 添加可视化跟踪功能
- 添加中文文档（README_CN.md）
- 添加示例代码目录（examples/）
  - 快速开始示例
  - 自适应策略示例
  - 多源/单源数据示例
  - 综合示例

### Fixed
- 修复工具模块中重复名称的 bug（logger => _logger）

## [0.2.1] - 2025-11-17
### Fixed
- 修复反投影的bug

## [Unreleased]

### Planned
- 完善单元测试
- 支持更多压缩策略
- 改进文档和教程
- 添加填充策略