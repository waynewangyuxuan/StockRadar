# StockRadar

StockRadar 是一个面向研究驱动、工程可扩展的量化策略执行与信号生成平台，模拟工业界 Quant Developer 和 Quant Researcher 的协作模式。

## 项目目标

StockRadar 旨在构建一个可扩展、模块化、适合策略研究和工程部署的量化信号系统，支持：

- 快速原型策略开发（Quant Research）
- 因子/指标插件式扩展（Quant Dev）
- 批量信号执行（Execution Engine）
- 后期可拓展到回测、实时流处理（Backtesting, Streaming）

## 系统架构

### 核心组件

1. **数据引擎**：Hive + HDFS（兼容 Parquet），负责统一存储原始行情、因子数据
2. **因子系统**：基于 FactorBase 抽象类，支持 Python 原型和 C++ 动态链接
3. **策略系统**：基于 StrategyBase 的策略框架，支持 YAML 配置和多策略组合
4. **调度与执行**：CLI + 批处理 Runner，可对接 Airflow / Prefect
5. **回测系统**：预留 Qlib 接入接口
6. **配置管理**：YAML 驱动的参数管理系统
7. **实时流处理**：预留 Kafka / Flink 接入接口

### 目录结构

```
StockRadar/
├── data/                # 原始数据目录（CSV / Parquet）
├── core/                # 核心框架（抽象类和主执行逻辑）
│   ├── factor_base.py
│   ├── strategy_base.py
│   └── runner.py
├── plugins/             # 插件式模块（因子 / 策略 / C++ 模块）
│   ├── factors/
│   ├── strategies/
│   └── cpp_modules/
├── configs/             # YAML 配置文件
├── jobs/                # 执行任务脚本
├── output/              # 信号输出
└── README.md
```

## 技术栈

- 数据处理：Hive + HDFS + Parquet
- 批量计算：PySpark / Pandas
- 插件机制：Python + C++ (pybind11)
- 配置管理：YAML
- 信号输出：CSV / JSON

## 系统特性

- 高可扩展性：因子、策略均可热插拔
- 高可维护性：解耦设计，便于重构
- 工程化程度高：贴近工业界量化系统架构
- 支持原型到生产迁移：从 Python 到 C++ 平滑演进

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置系统：
- 复制 `configs/template.yaml` 到 `configs/my_config.yaml`
- 修改配置参数

3. 运行示例：
```bash
python jobs/run_weekly_signal.py
```

## 开发指南

### 添加新因子

1. 在 `plugins/factors/` 下创建新的因子类
2. 继承 `FactorBase` 并实现必要方法
3. 在配置文件中注册因子

### 添加新策略

1. 在 `plugins/strategies/` 下创建新的策略类
2. 继承 `StrategyBase` 并实现必要方法
3. 在配置文件中配置策略参数

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License 