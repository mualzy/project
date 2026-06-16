# 查漏补缺执行记录

更新时间：2026-05-02 01:20 CST

## 1. 缺口清单与优先级

| 优先级 | 缺口 | 处理状态 | 处理理由 |
|---|---|---|---|
| 第一优先级 | 成功主实验缺少训练外独立 eval 表 | 已补齐 | 报告需要独立量化结果，不能只依赖训练日志 |
| 第一优先级 | 缺少推理效率统计 | 已补齐 | 论文/报告需要速度或工程效率视角 |
| 第一优先级 | 报告用图表分散在多个目录 | 已补齐 | 需要统一报告素材目录 |
| 第二优先级 | README 核对表缺失 | 已补齐 | 用户明确要求对照 README 审查 |
| 第二优先级 | 论文对齐说明缺失 | 已补齐 | 用户明确要求对照论文审查 |
| 第三优先级 | 初步实验报告缺失 | 已补齐 | 本轮核心交付 |
| 后续可选 | 预训练权重 release/torch.hub 验证 | 未补 | 当前自行训练权重已证明主链路，预训练对照不影响基础复现成立 |
| 后续可选 | Docker/W&B 在线功能 | 未补 | Windows/SSH 本地 venv 已稳定，在线记录不作为当前关键缺口 |

## 2. 本轮补做实验与分析

### 2.1 独立 checkpoint 评估

补做原因：`exp_110_full_adamw_amp` 已有训练日志和 checkpoint，但报告需要可复查的独立评估表。

命令摘要：

```powershell
.\.venv_cuda121\Scripts\python.exe scripts\evaluate_checkpoint.py `
  --model checkpoints\exp_110_full_adamw_amp\checkpoint_best.pth `
  --img-dir data\carvana_real\imgs `
  --mask-dir data\carvana_real\masks `
  --scale 0.125 `
  --validation 10 `
  --batch-size 4 `
  --num-workers 0 `
  --amp `
  --out results\tables\exp_110_eval_summary.csv
```

新增产物：

- `scripts/evaluate_checkpoint.py`
- `logs/exp_110_eval.log`
- `results/tables/exp_110_eval_summary.csv`

结果摘要：

- 验证集大小：508
- Dice：0.9913203716
- 运行时间：24.9994 秒
- 评估阶段峰值显存：443.59 MiB
- 状态：成功

### 2.2 推理效率统计

补做原因：当前实验已有预测图，但缺少单图推理速度和显存数据。

命令摘要：

```powershell
.\.venv_cuda121\Scripts\python.exe scripts\benchmark_inference.py `
  --model checkpoints\exp_110_full_adamw_amp\checkpoint_best.pth `
  --img-dir data\subsets\real_exp512\imgs `
  --scale 0.125 `
  --limit 20 `
  --warmup 3 `
  --out results\tables\exp_110_inference_benchmark.csv
```

新增产物：

- `scripts/benchmark_inference.py`
- `logs/exp_110_inference_benchmark.log`
- `results/tables/exp_110_inference_benchmark.csv`

结果摘要：

- 测试图像数：20
- 平均耗时：0.1859 秒/图
- 吞吐：5.3799 图/秒
- 推理阶段峰值显存：310.17 MiB
- 状态：成功

### 2.3 报告素材归档

补做原因：图表原本分散在 `data_summary/`、`results/` 下，报告引用需要统一路径。

新增目录：

- `reports_materials/figures/`
- `reports_materials/tables/`

整理内容：

- 数据样例图
- `exp_110` loss/metric 曲线
- `exp_110` 预测对比图
- `exp_102` 失败案例图
- `exp_106/exp_107` 对照曲线
- 实验登记表、指标汇总表、独立 eval 表、推理 benchmark 表

## 3. 未重复训练说明

本轮没有重复训练 `exp_110`，原因是该实验已经完整成功，且已有权重、日志、曲线、预测图和独立评估可复用。继续重复训练会增加时间成本和结果分散风险，不符合“优先复用已有成果”的原则。
