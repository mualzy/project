# README 复现核对表

更新时间：2026-05-02 01:20 CST  
对照来源：本地 `README.md` 与 GitHub `milesial/Pytorch-UNet` master 分支 README。

| README 项目 | 当前状态 | 证据路径 | 是否需要补做 | 备注 |
|---|---|---|---|---|
| 安装 CUDA | 已完成 | `env/hardware_info.txt`、`logs/exp_110_full_adamw_amp.log` | 否 | RTX 4070 Laptop GPU 可被 PyTorch CUDA 环境识别 |
| 安装 PyTorch >= 1.13 | 已完成 | `.venv_cuda121`、`env/dependency_fix_log.md` | 否 | 实际使用 torch 2.5.1+cu121/torchvision 0.20.1+cu121 |
| `pip install -r requirements.txt` | 部分完成 | `env/dependency_fix_log.md` | 否 | Windows/Python 3.12 下采用 CUDA 兼容环境与必要依赖修复，未机械使用最初坏环境 |
| `bash scripts/download_data.sh` 下载数据 | 部分完成 | `data/carvana_real/`、`data_summary/carvana_real/dataset_overview.md` | 否 | Bash/Kaggle 脚本在 Windows 下不作为主路径；实际通过 Kaggle 数据完成下载、解压和审计 |
| 数据目录为 `data/imgs` 与 `data/masks` | 部分完成 | `data/carvana_real/imgs`、`data/carvana_real/masks` | 否 | 为保留真实数据与子集，训练入口增加 `--img-dir/--mask-dir`，兼容原仓库结构 |
| 训练命令 `python train.py --amp` | 已完成 | `logs/exp_110_full_adamw_amp.log`、`checkpoints/exp_110_full_adamw_amp/` | 否 | 使用等价增强命令：AMP + GPU + 指定数据目录 + AdamW + best checkpoint |
| 训练参数 `--epochs --batch-size --learning-rate --scale --validation --amp` | 已完成 | `logs/run_history.csv`、`results/tables/experiment_registry.csv` | 否 | 所有关键参数均登记 |
| 保存 checkpoint | 已完成 | `checkpoints/exp_110_full_adamw_amp/checkpoint_best.pth` | 否 | 同时保留 epoch checkpoint |
| 预测命令 `python predict.py -i image.jpg -o output.jpg` | 已完成 | `logs/run_history.csv`、`results/predictions/exp_110_full_adamw_amp/` | 否 | 已对 6 个真实样本执行批量预测并保存 mask |
| 多图预测 | 已完成 | `results/predictions/exp_110_full_adamw_amp/` | 否 | README 支持多输入，本项目已批量验证 |
| 可视化输出 | 已完成 | `results/figures/prediction_samples/` | 否 | 已生成输入、真值、预测、误差对比面板 |
| 训练中评估 Dice | 已完成 | `logs/exp_110_full_adamw_amp.log` | 否 | 训练日志记录 validation Dice |
| 独立 eval | 已完成 | `results/tables/exp_110_eval_summary.csv`、`logs/exp_110_eval.log` | 否 | README 未单列 eval 命令，但仓库包含 `evaluate.py`，本轮补独立 checkpoint 评估 |
| W&B logging | 部分完成 | `logs/wandb_offline/`、`logs/exp_114_wandb_offline.log` | 用户配置后可同步 | Offline run 已验证；online 需要 W&B API key |
| 预训练模型 release / torch.hub | 已完成 | `results/tables/pretrained_release_check.csv`、`results/tables/torchhub_entry_check.csv`、`results/tables/exp_113_pretrained_release_scale05_eval512.csv` | 否 | GitHub release v3.0 scale0.5 权重已下载、torch.hub 入口已验证、预测和评估 |
| Dockerfile | 部分完成 | `logs/run_history.csv`、`notes/manual_setup_required.md` | 是 | Docker CLI 存在，但 Docker Desktop Linux engine 未运行 |

## 结论

README 中最核心的三条链路已经真实完成：数据准备、GPU 训练、预测输出。环境安装和数据下载没有完全照搬 README 命令，而是按 Windows/SSH 环境进行了兼容性调整，并保留了记录。未验证项主要是 Docker、W&B 在线记录和预训练 release/torch.hub，这些不影响当前自行训练复现的成立。
