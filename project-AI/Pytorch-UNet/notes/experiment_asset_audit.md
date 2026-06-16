# 实验资产审计文档

更新时间：2026-05-02 01:20 CST  
审计范围：`D:\project-AI\Pytorch-UNet` 当前工作目录、实验日志、权重、数据摘要、图表、报告素材。

## 1. 总体结论

当前项目已经具备“基础复现可报告”的核心资产：真实 Carvana 数据集、可用 CUDA 环境、完整训练日志、成功的全量数据基线权重、独立评估结果、推理结果、训练曲线、预测对比图、失败案例记录和实验登记表均已落盘。

最可靠的主实验为 `exp_110_full_adamw_amp`：在 5088 对真实 Carvana 图像/掩码上训练 2 个 epoch，使用 RTX 4070 Laptop GPU、AMP、AdamW、batch size 4、scale 0.125，最终独立验证 Dice 为 `0.9913203716`。该结果可作为当前报告中的主基线。

## 2. 代码与配置资产

### 已有内容

| 类别 | 路径 | 状态 | 说明 |
|---|---|---|---|
| 原始训练入口 | `train.py` | 已修改 | 保留主训练逻辑，增加工程参数与安全保护 |
| 原始预测入口 | `predict.py` | 未改主逻辑 | 已用于生成真实样本预测图 |
| 原始评估入口 | `evaluate.py` | 原仓库保留 | 训练中调用；本轮另补独立 checkpoint 评估脚本 |
| 模型定义 | `unet/` | 原仓库保留 | U-Net contracting/expanding path 结构 |
| 数据读取 | `utils/data_loading.py` | 已修改 | Windows 下改为顺序扫描 mask，避免 multiprocessing/Pickle 风险 |
| 依赖记录 | `requirements.txt`、`env/dependency_fix_log.md` | 已记录 | 实际可用环境为 `.venv_cuda121` |
| 运行脚本 | `scripts/*.py` | 已补充 | 数据审计、子集构建、曲线绘制、预测面板、独立评估、推理 benchmark |

### 源码修改摘要

| 文件 | 修改点 | 修改原因 | 风险 |
|---|---|---|---|
| `train.py` | 增加 `--num-workers`、`--checkpoint-dir`、`--img-dir`、`--mask-dir`、`--save-best`、`--abort-on-nonfinite`、`--optimizer` | Windows/SSH 环境下需要稳定数据加载、可控输出目录、失败保护和 AdamW 修复实验 | 最小侵入；默认参数保持原仓库行为为主 |
| `utils/data_loading.py` | Windows 下 mask 唯一值扫描由 `Pool` 改为顺序扫描 | 避免 Windows 多进程序列化与远程 SSH 兼容问题 | 仅影响初始化速度，不改变样本语义 |
| `.gitignore` | 忽略本地虚拟环境 | 避免大体积环境目录进入版本控制 | 无 |
| `scripts/evaluate_checkpoint.py` | 新增独立评估脚本 | 补齐训练外 checkpoint 评估证据 | 辅助脚本，不改主逻辑 |
| `scripts/benchmark_inference.py` | 新增推理速度脚本 | 补齐效率分析证据 | 辅助脚本，不改主逻辑 |

## 3. 模型与训练产物

### 可直接复用的主产物

| 实验 | 状态 | 关键产物 |
|---|---|---|
| `exp_110_full_adamw_amp` | 成功 | `checkpoints/exp_110_full_adamw_amp/checkpoint_best.pth`、训练日志、曲线、预测图、独立评估表 |
| `exp_107_real512_amp_lr1e5` | 成功 | 512 样本 AMP 扩展实验，适合用于效率/稳定性对比 |
| `exp_106_real512_noamp_lr1e5` | 成功 | 512 样本非 AMP 对照 |
| `exp_108_real512_scale025_amp` | 成功 | 输入 scale 扩展实验 |
| `exp_102_real_baseline_amp` | 失败但有价值 | RMSprop + AMP 全量训练 NaN 失败案例，可用于工程修复分析 |
| `exp_105_real_baseline_amp_lr1e5` | 失败但有价值 | Epoch 1 达到高 Dice 后 Epoch 2 NaN，说明优化器/数值稳定性问题 |

### 权重状态

- 最佳可用权重：`checkpoints/exp_110_full_adamw_amp/checkpoint_best.pth`
- 最终 epoch 权重：`checkpoints/exp_110_full_adamw_amp/checkpoint_epoch1.pth`、`checkpoint_epoch2.pth`
- 失败实验权重：保留在 `checkpoints/exp_102_real_baseline_amp`、`checkpoints/exp_105_real_baseline_amp_lr1e5`，仅用于诊断，不建议作为报告主权重。

## 4. 日志与表格资产

| 文件 | 状态 | 说明 |
|---|---|---|
| `logs/run_history.csv` | 已维护 | 记录关键命令、状态、结果摘要 |
| `logs/issue_log.md` | 已维护 | 记录 Kaggle、CUDA、NaN、Windows 兼容等问题 |
| `logs/exp_110_full_adamw_amp.log` | 已存在 | 主实验训练日志 |
| `logs/exp_110_eval.log` | 已存在 | 本轮补做的独立评估日志 |
| `logs/exp_110_inference_benchmark.log` | 已存在 | 本轮补做的推理效率日志 |
| `results/tables/experiment_registry.csv` | 已维护 | 实验总登记表 |
| `results/tables/metrics_summary.csv` | 已更新 | 汇总关键指标，含本轮 eval/benchmark |
| `results/tables/exp_110_eval_summary.csv` | 已新增 | `exp_110` 独立评估结果 |
| `results/tables/exp_110_inference_benchmark.csv` | 已新增 | `exp_110` 推理速度统计 |

## 5. 可视化与结果图资产

| 类别 | 路径 | 状态 |
|---|---|---|
| 数据样例 | `data_summary/carvana_real/visual_checks/` | 已生成 |
| 训练曲线 | `results/figures/training_curves/` | 已生成 |
| 预测样例 | `results/figures/prediction_samples/` | 已生成 |
| 失败案例 | `results/figures/failure_cases/` | 已生成 |
| 报告用图片副本 | `reports_materials/figures/` | 本轮整理完成 |
| 预测 mask 输出 | `results/predictions/exp_110_full_adamw_amp/` | 已生成 |

## 6. 数据与样本资产

| 项目 | 当前状态 |
|---|---|
| 数据集 | Kaggle Carvana Image Masking Challenge 训练集 |
| 图像目录 | `data/carvana_real/imgs` |
| 掩码目录 | `data/carvana_real/masks` |
| 有效配对数 | 5088 |
| 图像尺寸 | 1918 x 1280 |
| 掩码值 | 0/1 二值 mask |
| 异常样本 | 0 |
| 数据审计文档 | `data_summary/carvana_real/dataset_overview.md` |
| 样本统计表 | `data_summary/carvana_real/sample_stats.csv` |

## 7. 已有文档资产

| 文件 | 状态 | 用途 |
|---|---|---|
| `README_agent.md` | 已存在 | 工程接管与阶段摘要入口 |
| `notes/project_intake.md` | 已存在 | 仓库接管说明 |
| `notes/stage_summary.md` | 已维护 | 阶段总结 |
| `notes/experiment_notes.md` | 已维护 | 实验记录 |
| `notes/smoke_test_summary.md` | 已存在 | Smoke test 总结 |
| `reports_materials/reusable_text_snippets.md` | 已存在 | 报告可复用文字 |

## 8. 缺失或部分完成内容

| 缺口 | 状态 | 当前处理 |
|---|---|---|
| 原仓库 Docker 流程 | 未验证 | 本机 Windows + venv 已满足复现需求，Docker 不是当前优先项 |
| W&B 在线记录 | 部分完成 | 为避免交互和网络不确定性，使用本地日志和曲线替代 |
| 预训练权重 release / torch.hub 验证 | 未验证 | 当前以自行训练权重为主；后续可补做预训练权重推理对照 |
| 原论文 ISBI 生物医学 benchmark | 未复现 | 当前仓库面向 Carvana，不提供原论文完整数据与训练协议 |
| 高分辨率 scale 0.5/1.0 全量长训 | 未完成 | 受 8GB 显存、16GB 内存和时间限制，本阶段采用 scale 0.125 稳妥基线 |

## 9. 建议补充内容与本轮处理结果

本轮已补齐两个最关键缺口：

1. 独立 checkpoint 评估：`results/tables/exp_110_eval_summary.csv`
2. 推理效率统计：`results/tables/exp_110_inference_benchmark.csv`

其余缺口不影响当前“基础复现成立”的判断，建议放入后续工作而不是在本轮继续消耗训练时间。
