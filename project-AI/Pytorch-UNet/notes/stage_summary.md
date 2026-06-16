# Stage Summary

## 2026-04-30 17:21 CST - Stage 1 Repository Intake

- Completed: cloned `milesial/Pytorch-UNet` master and reviewed README, training, prediction, dataset, model, metric, and download logic.
- Current blockers: none for code intake.
- Key result: repository is compact and runnable with `train.py` and `predict.py`; data must be flat in `data/imgs` and `data/masks`.
- Next: environment detection and dependency setup.

## 2026-04-30 18:50 CST - Stage 2 Environment Setup

- Completed: created `.venv`, installed CPU PyTorch stack and support dependencies, verified torch import.
- Current blockers: CUDA PyTorch install timed out; GPU cannot be used by PyTorch yet.
- Key result: CPU runtime is usable; `nvidia-smi` sees RTX 4070 Laptop GPU.
- Next: generate or download data, audit dataset, then run smoke test.

## 2026-04-30 18:55 CST - Stage 3 Data Preparation

- Completed: detected missing Kaggle credentials, generated 48 synthetic Carvana-compatible image/mask pairs, audited file pairing and image sizes.
- Current blockers: original Carvana dataset still requires Kaggle credentials.
- Key result: `data_summary/dataset_overview.md`, `data_summary/sample_stats.csv`, and visual panels in `data_summary/visual_checks/` were generated.
- Next: run `exp_001_smoke` training and prediction chain.

## 2026-04-30 19:15 CST - Stages 4-6 Experiments

- Completed: smoke test, baseline training, prediction, curve generation, and two extension experiments.
- Current blockers: original Carvana benchmark and GPU/AMP experiments remain blocked by missing Kaggle credentials and CUDA wheel install timeout.
- Key result: best current synthetic run is `exp_002_baseline_cpu_synth` with max validation Dice 0.9629.
- Next: when credentials/GPU wheel are available, rerun the same scripts on real Carvana data with CUDA and AMP.

## 2026-05-01 - Resume With Manual Configuration

- Completed: verified Kaggle config file exists and Kaggle CLI can list Carvana files; installed CUDA PyTorch and verified RTX 4070 Laptop GPU via `torch.cuda`.
- Current blockers: after successful real data download and one GPU smoke, PyTorch import/training became unstable and repeatedly hung. Reboot is recommended before continuing.
- Key result: real Carvana data downloaded and audited: 5088 image/mask pairs, zero anomalies, 1918x1280.
- Next: reboot/reset host, then continue real GPU AMP baseline and extension experiments.

## 2026-05-01 - Real Data Download and Audit

- Completed: downloaded `train_hq.zip` and `train_masks.zip`, extracted to `data/carvana_real/imgs` and `data/carvana_real/masks`.
- Completed: audited full dataset into `data_summary/carvana_real`; 5088 valid pairs, zero anomalies.
- Completed: created 64-sample real subset at `data/subsets/real_smoke64`.
- Partial: one GPU smoke run reached checkpoint save at `checkpoints/exp_101_real_smoke/checkpoint_epoch1.pth`, but follow-up runs exposed environment instability.

## 2026-05-01 - Real GPU Reproduction After Reboot

- Completed: verified `.venv_cuda121` with `torch 2.5.1+cu121`, CUDA available, RTX 4070 Laptop GPU detected.
- Completed: reran real-data GPU smoke as `exp_101_real_smoke_clean`; training, checkpoint save, prediction, and panels all succeeded.
- Completed: attempted two full 5088-pair AMP baselines, `exp_102` and `exp_105`; both exposed repeatable epoch-2 `nan` instability and were recorded as failed diagnostic runs.
- Completed: created 512-pair subset and ran three controlled real-data experiments: non-AMP baseline `exp_106`, AMP extension `exp_107`, and scale 0.25 extension `exp_108`.
- Key result: best stable real-data result is `exp_107_real512_amp_lr1e5`, max validation Dice 0.9443, final validation Dice 0.8947, runtime 89.69 seconds.
- Outputs: checkpoints, logs, training curves, prediction masks, prediction panels, failure panels, experiment registry, and metrics summary were written under project directories.
- Current blockers: full 5088-pair 2-epoch training is not yet stable; it needs a safer long-run recipe such as lower effective LR, gradient clipping, scheduler adjustment, or best-checkpoint protection before claiming full-dataset convergence.
- Next: if more runtime is allowed, run full dataset from the stable `exp_107` recipe with best-model save and early abort on non-finite loss.

## 2026-05-02 - Full-Dataset Failure Repair

- Completed: added guarded training controls to `train.py`: `--save-best`, `--abort-on-nonfinite`, and `--optimizer {rmsprop,adamw,sgd}`.
- Completed: diagnosed `exp_105` epoch1 checkpoint; tensors were finite but weight magnitudes were excessive, so resume repair was not reliable.
- Completed: `exp_109_full_resume_amp_lr1e6` intentionally tested resume repair and aborted at step 5 on non-finite loss, preventing a corrupt checkpoint.
- Completed: `exp_110_full_adamw_amp` trained the full 5088-pair real dataset for 2 epochs with AdamW + AMP and completed successfully.
- Key result: full-data stable run reached max/final validation Dice 0.9913 and saved `checkpoints/exp_110_full_adamw_amp/checkpoint_best.pth`.
- Outputs: full-data curves, best checkpoint, epoch checkpoints, prediction masks, comparison panels, updated experiment registry, metrics summary, run history, and issue log.
- Current blockers: none for a runnable full-data reproduction. Remaining scientific limitation is that this is a short 2-epoch reproduction at scale 0.125, not a leaderboard-grade Carvana training recipe.
## 2026-05-02 01:20 CST - 复现审查、查漏补缺与初步报告

### 本阶段完成内容
- 完成当前实验资产全面盘点，新增 `notes/experiment_asset_audit.md`。
- 对照仓库 README 完成复现核对，新增 `notes/readme_reproduction_checklist.md`。
- 对照 U-Net 论文完成覆盖范围分析，新增 `notes/paper_alignment_check.md`。
- 补齐主实验 `exp_110_full_adamw_amp` 的独立 checkpoint 评估，输出 `results/tables/exp_110_eval_summary.csv`。
- 补齐推理效率 benchmark，输出 `results/tables/exp_110_inference_benchmark.csv`。
- 统一整理报告图片和表格到 `reports_materials/figures` 与 `reports_materials/tables`。
- 完成详细初步实验报告 `reports_materials/initial_experiment_report.md`。

### 当前阻塞点
- 无基础复现阻塞。
- 未验证项主要是预训练 release/torch.hub、Docker、W&B 在线记录，以及原论文 ISBI 生物医学 benchmark。

### 关键结果
- `exp_110_full_adamw_amp` 独立评估 Dice：0.9913203716。
- 推理 benchmark：0.1859 秒/图，5.3799 图/秒。
- 当前结论：仓库主线在 Carvana 任务上的基础复现完整成立；原论文全量 benchmark 严格复刻未完成。

### 下一步计划
- 如需进一步增强报告，可补做预训练模型加载和推理对照。
- 如需提升指标可信度，可尝试更高 scale 或更长训练，但需重新评估显存和时间成本。
## 2026-05-02 09:50 CST - ISBI、预训练、W&B、Docker 补充检查

### 本阶段完成内容
- 下载并审计 Kaggle ISBI-2012 数据副本，train/test 各 30 对 512x512 图像与标签，异常为 0。
- 为灰度生物医学图像补充 `--channels` 参数，覆盖训练、预测和独立评估脚本。
- 完成 `exp_201_isbi2012_smoke` 灰度链路验证。
- 完成 `exp_202_isbi2012_baseline_adamw_amp` 10 epoch 本地 benchmark，test Dice 为 0.9104232788。
- 下载并验证 GitHub release v3.0 官方 `scale0.5` 预训练权重，512 样本子集 Dice 为 0.991784。
- 验证 W&B offline 可用；确认 online 需要用户配置 API key。
- 检查 Docker 状态；确认 Docker CLI 存在但 Docker Desktop Linux engine 未运行。
- 新增 `notes/manual_setup_required.md`，说明用户需要手动配置的内容。

### 当前阻塞点
- Docker：需要用户启动 Docker Desktop 并启用 Linux/WSL2 backend。
- W&B online：需要用户登录 W&B 并配置 API key。
- 原论文严格官方复刻：仍需要对应 ISBI Cell Tracking Challenge 2015 数据、官方评测工具和原协议。

### 关键结果
- ISBI-2012 local test Dice：0.9104232788。
- 官方预训练 Carvana 512 子集 Dice：0.991784。
- W&B offline run：`logs/wandb_offline/wandb/offline-run-20260502_092937-tn5vd4jh`。

### 下一步计划
- 用户配置 Docker/W&B 后，可补做 Docker build/run 和 W&B online sync。
- 若要进一步严格复刻原论文，需要单独下载 Cell Tracking Challenge 数据并按官方指标评估。

## 2026-05-02 10:40 CST - Docker/W&B 配置复查

### 本阶段完成内容
- 复查 Docker Desktop：daemon 已可用，Linux engine 正常。
- 复查 Docker GPU：`nvidia/cuda:12.4.1-base-ubuntu22.04` 容器内 `nvidia-smi` 成功看到 RTX 4070 Laptop GPU。
- 复查 W&B online：仍失败，提示 `_netrc` 中凭据存在但 API key verification failed / user is not logged in。
- 尝试原仓库 Dockerfile build：基础镜像元数据可读取，build context 经 `.dockerignore` 缩小到约 122 KB，但 pip 安装依赖时 PyPI/NGC HTTPS SSLEOF，构建失败。

### 当前阻塞点
- W&B online 需要重新 `wandb login --relogin` 并使用有效 API key。
- Dockerfile build 需要修复 Docker build 容器内 PyPI/NGC HTTPS 访问，或使用可用 pip 镜像/代理/wheelhouse。

### 关键结果
- Docker runtime/GPU 配置正常。
- Dockerfile build 未完全成功，失败点是依赖下载网络层。
- W&B offline 可用，online 不可用。

### 下一步计划
- 用户修复 W&B API key 后可补在线同步/在线实验记录。
- 用户提供 pip 镜像或 Docker build 网络修复后可继续完成 Docker image build 和容器内 smoke test。

## 2026-05-02 10:55 CST - Docker build 网络修复与容器推理验证

### 本阶段完成内容
- 按 pip 镜像优先策略修改 Dockerfile：依赖安装使用 `https://pypi.tuna.tsinghua.edu.cn/simple`。
- 删除 Dockerfile 中不必要的 pip 自升级步骤，减少构建阶段外网访问点。
- 使用 `docker build --no-cache --progress=plain -t my-unet .` 成功构建镜像。
- 使用 `docker run --rm --gpus all my-unet ...` 验证容器内 PyTorch、CUDA 和项目 `unet` 导入。
- 挂载数据、结果目录和官方预训练权重，完成容器内 `predict.py` 推理并保存 mask。

### 当前阻塞点
- Docker build 已不阻塞。
- W&B online 状态本轮未重新处理，用户已声明解决；后续可单独复查或直接运行在线训练记录。

### 关键结果
- Docker image：`my-unet:latest`。
- 镜像大小：约 27.7GB。
- 容器内 CUDA：可用，GPU 为 RTX 4070 Laptop GPU。
- 容器预测输出：`results/predictions/exp_116_docker_pretrained_smoke/00087a6bd4dc_01.png`。

### 下一步计划
- 可将 Docker 命令整理进报告附录。
- 若要减少镜像体积，可另建 lighter Dockerfile，但这会偏离原仓库 NGC PyTorch 基础镜像。
