# 论文对齐检查说明

更新时间：2026-05-02 01:20 CST  
论文：U-Net: Convolutional Networks for Biomedical Image Segmentation  
论文入口：https://arxiv.org/abs/1505.04597

## 1. 论文核心目标与模型定位

U-Net 论文面向生物医学图像分割，目标是在训练图像数量有限的情况下，通过端到端全卷积网络获得精确的像素级分割结果。其模型定位不是分类网络，而是输入图像到输出分割 mask 的 dense prediction 模型。

论文强调的核心设计包括：

- contracting path 提取上下文语义；
- expanding path 恢复空间分辨率；
- skip connection 将浅层定位信息传到解码端；
- 数据增强，尤其是 elastic deformation；
- overlap-tile strategy，用于处理大尺寸图像；
- 快速、端到端的分割推理。

## 2. 当前仓库与论文关系

`milesial/Pytorch-UNet` 是 U-Net 思想的 PyTorch 工程实现，但其 README 和默认数据流程面向 Kaggle Carvana Image Masking Challenge，不是原论文 ISBI 生物医学挑战的完整官方复刻。因此，当前实验属于“U-Net 方法在 Carvana 二值分割任务上的工程复现”，而不是对论文中所有生物医学 benchmark 的严格复现。

## 3. 当前实验覆盖情况

| 论文维度 | 当前覆盖状态 | 证据 | 说明 |
|---|---|---|---|
| U-Net 编码器-解码器结构 | 已覆盖 | `unet/unet_model.py`、`unet/unet_parts.py` | 仓库实现标准 U-Net 结构 |
| 像素级二值分割 | 已覆盖 | `results/predictions/exp_110_full_adamw_amp/` | Carvana 前景/背景 mask 预测 |
| 端到端训练 | 已覆盖 | `logs/exp_110_full_adamw_amp.log` | 从随机初始化训练至有效 Dice |
| Dice 评价 | 已覆盖 | `results/tables/exp_110_eval_summary.csv` | 独立验证 Dice 为 0.9913203716 |
| 定性分割可视化 | 已覆盖 | `results/figures/prediction_samples/` | 输入、真值、预测、误差面板 |
| 快速推理分析 | 部分覆盖 | `results/tables/exp_110_inference_benchmark.csv` | 在 scale 0.125 下统计推理速度 |
| 少样本/数据增强论证 | 部分覆盖 | `exp_106`、`exp_107`、`exp_108` | 有小子集扩展，但未复刻论文增强策略 |
| 生物医学 ISBI benchmark | 未覆盖 | 无 | 当前仓库数据和任务不是 ISBI |
| elastic deformation | 未覆盖 | 无 | 当前未加入论文同款增强 |
| overlap-tile strategy | 未覆盖 | 无 | 当前预测脚本按 resize/整图推理工作 |

## 4. 当前复现层级判断

| 层级 | 状态 | 说明 |
|---|---|---|
| 代码能跑 | 已达到 | Windows/SSH/CUDA 环境可运行 |
| Demo 可展示 | 已达到 | 真实样本预测结果已保存 |
| 基线训练完成 | 已达到 | `exp_110_full_adamw_amp` 完整成功 |
| 评估结果完整 | 已达到基础要求 | 独立 eval 表已补齐 |
| 与论文主张相符的分析 | 部分达到 | 体现了 U-Net 对二值分割的有效性，但未严格复刻生物医学数据与论文实验协议 |
| 论文级严格复刻 | 未达到 | 缺 ISBI 数据、原论文增强、原论文挑战指标和训练协议 |

## 5. 结论

当前结果足以支持“目标仓库在 Carvana 任务上的基础复现成立”：训练链路、推理链路、评估链路、图像展示和工程记录均完整。需要明确的是，这不是对 U-Net 原论文全部实验表格的严格复现，而是对该 PyTorch 仓库主线能力和 U-Net 分割思想的可运行复现与工程化整理。

## 6. 2026-05-02 补充：ISBI-2012 本地 benchmark 尝试

本轮进一步下载并跑通了 Kaggle 上的 ISBI-2012 EM segmentation 数据副本：

- 数据集链接：https://www.kaggle.com/datasets/hamzamohiuddin/isbi-2012-challenge
- 训练集：30 对 512x512 图像/标签
- 测试集：30 对 512x512 图像/标签
- 审计结果：train/test 异常均为 0

为适配灰度生物医学图像，工程上为 `train.py`、`predict.py`、`scripts/evaluate_checkpoint.py` 增加了 `--channels` 参数，默认值仍为 3，不影响 Carvana RGB 实验。

补充实验结果：

| 实验 | 数据 | 配置 | 结果 |
|---|---|---|---|
| `exp_201_isbi2012_smoke` | ISBI train 30 pairs | 1 epoch, channels=1, scale=1.0 | max val Dice 约 0.8726 |
| `exp_202_isbi2012_baseline_adamw_amp` | ISBI train 30 pairs, local test 30 labels | 10 epochs, channels=1, AdamW, AMP | max val Dice 约 0.9389，local test Dice 0.9104232788 |

这一步将复现范围从 Carvana 扩展到了生物医学灰度分割数据，显著增强了论文对齐程度。但仍需说明：这仍不是原论文所有 ISBI Cell Tracking Challenge 2015 项目的严格官方提交复刻，也没有使用官方 challenge evaluation 工具。
