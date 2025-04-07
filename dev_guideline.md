# File Structure
StockRadar/
├── data/                          
│
├── data_fetcher/                  
│   ├── base.py                    
│   ├── yfinance_provider.py       
│   ├── simulator_provider.py      
│   ├── hive_provider.py           
│   ├── api_provider_template.py   
│   └── utils/                     
│       ├── cache.py
│       └── validators.py
│
├── data_processor/                
│   ├── cleaner.py                 
│   ├── transformer.py             
│   └── validator.py               
│
├── data_storage/                  
│   ├── local_storage.py           
│   ├── timescaledb_storage.py     
│   ├── redis_cache.py             
│   └── version_control.py         
│
├── core/                          
│   ├── factor_base.py             
│   ├── strategy_base.py           
│   └── runner.py                  
│
├── plugins/                       
│   ├── factors/                   
│   │   ├── ma_factor.py
│   │   └── volume_spike_factor.py
│   ├── strategies/                
│   │   ├── golden_cross.py
│   │   ├── mean_reversion.py
│   │   └── momentum_breakout.py
│   └── cpp_modules/               
│       └── README.md              
│
├── strategy_engine/               
│   ├── base.py                    
│   ├── registry.py                
│   ├── ensemble.py                
│   ├── schema.py                  
│   └── evaluator.py               
│
├── configs/                       
│   └── config.yaml                
│
├── jobs/                          
│   └── run_weekly_signal.py       
│
├── backtester/                    
│   ├── simulator.py               
│   ├── evaluator.py               
│   └── metrics.py                 
│
├── output/                        
├── logs/                          
├── README.md                      
├── requirements.txt               
├── .gitignore                     
└── Dockerfile

## 说明：
	•	data_fetcher/ 完全解耦数据源，未来可拆成微服务
	•	data_processor/ 负责清洗 & 计算所有因子
	•	data_storage/ 持久化 + 缓存 + 版本管理
	•	core/ 提供抽象接口，策略/因子/Runner 全部基于它
	•	plugins/ 放所有 Python 因子/策略 & C++ 插件
	•	strategy_engine/（可选）做策略组合 & 回测评估
	•	configs/ YAML 驱动全流程参数
	•	jobs/ 一行命令跑全流水线
	•	backtester/ 信号效果验证


# Strategy Engine 模块概述

## 1. 模块目标
策略系统负责把“行情+因子”→“买/卖/持有信号”。

## 2. 工业界设计参考
| 组成部分           | 工业实践描述                                     |
|--------------------|--------------------------------------------------|
| 策略类接口（Base） | 抽象基类，定义 `generate_signals()` 等统一接口   |
| 策略注册机制        | 可热插拔、动态加载，支持策略替换与版本管理       |
| 参数化配置         | 所有超参数从 YAML/JSON 加载，便于实验对比         |
| 策略组合器（Ensemble） | 多策略加权、投票或择优合成最终信号            |
| 策略输出规范       | 统一输出格式，便于评估、回测和可视化             |
| 策略评估机制       | 记录性能指标（收益、胜率、夏普比率等）           |

## 3. 模块结构设计
├── strategy_engine/               
│   ├── base.py                    
│   ├── registry.py                
│   ├── ensemble.py                
│   ├── schema.py                  
│   └── evaluator.py 

## 4. 接口设计建议
- **StrategyBase**（抽象）
- **GoldenCrossStrategy** 示例

## 5. 信号结构定义（schema.py）
```json
{
  "ticker": "AAPL",
  "date": "2024-04-01",
  "signal_type": "BUY",
  "value": 1,
  "confidence": 0.87,
  "source": "golden_cross"
}

---

# Moduel Interaction

1. Entry Point：jobs/run_weekly_signal.py

作用：脚本层入口，一行命令触发整条流水线。

调用：创建并调用 StrategyRunner.run()。

2. 核心调度：core/runner.py

职责：读取配置、加载数据、调度因子和策略、保存信号、记录监控指标。

依赖：

data_fetcher 提供原始数据。

data_processor 进行清洗和预处理。

plugins/factors 计算各类因子。

plugins/strategies 生成交易信号。

data_storage 持久化或缓存数据与信号。

3. 数据获取层：data_fetcher

职责：提供统一接口 DataProviderBase.fetch()，支持多种数据源。

输出：标准化的 DataFrame（包含 ticker,date,open,high,low,close,volume）。

交互：被 runner 调用，返回原始行情数据给 data_processor。

4. 数据处理层：data_processor

职责：

cleaner 清洗：缺失值填充、异常值剔除、格式校验。

transformer 特征：计算基础指标（收益率、VWAP）、技术指标（MA、波动率）。

输出：带有因子列的 DataFrame。

交互：接收 raw_df，输出 proc_df 给因子模块或直接给策略。

5. 因子插件：plugins/factors

职责：基于 FactorBase 抽象，实现各类因子计算。

示例：MovingAverage, VolumeSpike。

交互：runner 根据配置动态加载，调用 calculate(data)，生成 factors_df。

6. 策略插件：plugins/strategies

职责：基于 StrategyBase 抽象，实现信号生成逻辑。

示例：GoldenCrossStrategy, MeanReversionStrategy。

交互：runner 加载并传入 proc_df + factors_df，调用 generate_signals()，得到 signals_df。

7. 数据存储层：data_storage

职责：持久化行情、因子、信号，支持 Redis 缓存与 TimescaleDB 持久化。

交互：

runner 将 signals_df 保存到文件系统或数据库。

未来可供 backtester 或实时服务读取。

8. 策略框架增强：strategy_engine

职责：提供策略注册、组合（Ensemble）、评估（Evaluator）等高级功能。

交互：可由 runner 或专门脚本调用，用于多策略融合和性能评估。

9. 回测系统（可选）：backtester

职责：读取 signals_df 和行情数据，模拟交易并计算绩效指标。

交互：调用 data_storage 获取历史数据，使用 signals_df 驱动持仓逻辑。

模块间调用时序简述

启动：run_weekly_signal.py → StrategyRunner.run()

拉数据：runner._load_data() → data_fetcher → raw_df

清洗：data_processor.cleaner.clean(raw_df) → clean_df

特征：data_processor.transformer.transform(clean_df) → proc_df

因子计算：循环 plugins/factors，调用 calculate(proc_df) → factors_df

策略生成：循环 plugins/strategies，调用 generate_signals(proc_df, factors_df) → signals_df

持久化：data_storage.save(signals_df) + 可选缓存

结束：runner 记录监控指标与血缘信息，脚本退出

以上即为 StockRadar 各模块的典型交互流程，便于团队协作与后续扩展。

