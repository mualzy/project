# 需要用户手动下载或配置的内容

更新时间：2026-05-02 09:45 CST

## 1. 当前不需要你再手动配置的内容

以下内容已经由本地环境自动完成：

| 内容 | 当前状态 | 位置 |
|---|---|---|
| Carvana Kaggle 数据 | 已下载并审计 | `data/carvana_real/` |
| ISBI-2012 Kaggle 数据副本 | 已下载并审计 | `data/isbi2012_raw/` |
| 官方预训练 scale0.5 权重 | 已下载并验证 | `C:\Users\admin\.cache\torch\hub\checkpoints\unet_carvana_scale0.5_epoch2.pth` |
| W&B offline 本地记录 | 已验证 | `logs/wandb_offline/` |

## 2. W&B 在线记录配置

当前状态：`wandb` 包已安装，offline run 已成功创建。用户配置后再次检查发现 `C:\Users\admin\_netrc` 中有凭据，但 W&B API key verification failed，online run 仍失败。

你需要手动做：

1. 打开 W&B API key 页面：  
   https://wandb.ai/authorize
2. 登录你的 W&B 账号，复制 API key。
3. 在 PowerShell 中强制重新登录：

```powershell
cd D:\project-AI\Pytorch-UNet
.\.venv_cuda121\Scripts\wandb.exe login --relogin
```

4. 按提示粘贴 API key。
5. 验证：

```powershell
.\.venv_cuda121\Scripts\wandb.exe status
```

如果要把本轮已经生成的 offline run 上传到云端，执行：

```powershell
.\.venv_cuda121\Scripts\wandb.exe sync logs\wandb_offline\wandb\offline-run-20260502_092937-tn5vd4jh
```

也可以设置环境变量方式：

```powershell
$env:WANDB_API_KEY="你的_API_KEY"
$env:WANDB_MODE="online"
```

## 3. Docker 配置

当前状态：Docker Desktop 已恢复，`docker info` 正常，`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 已验证容器内可见 RTX 4070 Laptop GPU。

仍需注意：

1. 本仓库原始 Dockerfile 使用 `nvcr.io/nvidia/pytorch:22.11-py3`，基础镜像体积很大。
2. 早前 `docker build -t pytorch-unet:agent-check .` 失败在容器内 pip 访问 PyPI/NGC 的 SSL EOF，不是 Docker daemon 或 GPU runtime 问题。
3. 已新增 `.dockerignore`，避免把数据、权重、虚拟环境和结果目录发送进 build context。
4. 已通过 Dockerfile 中的 pip 镜像配置完成 build：`docker build --no-cache --progress=plain -t my-unet .`。

参考链接：

- Docker Desktop 下载与安装：  
  https://www.docker.com/products/docker-desktop/
- Docker Desktop WSL2 后端说明：  
  https://docs.docker.com/desktop/features/wsl/
- NVIDIA CUDA on WSL：  
  https://docs.nvidia.com/cuda/wsl-user-guide/index.html

已通过的验证：

```powershell
docker info
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

已成功的构建命令：

```powershell
cd D:\project-AI\Pytorch-UNet
docker build --no-cache --progress=plain -t my-unet .
```

已成功的容器推理命令摘要：

```powershell
docker run --rm --gpus all `
  -v ${PWD}/data:/workspace/unet/data `
  -v ${PWD}/results:/workspace/unet/results `
  -v C:/Users/admin/.cache/torch/hub/checkpoints:/workspace/checkpoints `
  my-unet python predict.py `
  --model /workspace/checkpoints/unet_carvana_scale0.5_epoch2.pth `
  --scale 0.5 `
  --input data/subsets/real_exp512/imgs/00087a6bd4dc_01.jpg `
  --output results/predictions/exp_116_docker_pretrained_smoke/00087a6bd4dc_01.png
```

注意：本仓库 `Dockerfile` 基础镜像是 `nvcr.io/nvidia/pytorch:22.11-py3`，首次拉取可能需要数 GB 下载空间和较长时间。若 NGC 拉取受限，需要按 NVIDIA NGC 文档配置访问。

## 4. 官方预训练权重

当前已自动下载并验证 `scale0.5` 权重，不需要你手动下载。若后续要手动下载，链接如下：

- scale 0.5：  
  https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth
- scale 1.0：  
  https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth

建议放置位置：

```text
C:\Users\admin\.cache\torch\hub\checkpoints\
```

也可以放在项目内：

```text
D:\project-AI\Pytorch-UNet\checkpoints\pretrained\
```

使用示例：

```powershell
.\.venv_cuda121\Scripts\python.exe predict.py `
  --model C:\Users\admin\.cache\torch\hub\checkpoints\unet_carvana_scale0.5_epoch2.pth `
  --scale 0.5 `
  --input data\subsets\real_exp512\imgs\00087a6bd4dc_01.jpg `
  --output results\predictions\pretrained_demo.png
```

## 5. 原论文 ISBI / 生物医学 benchmark 数据

本轮已经下载并跑通 Kaggle 上的 ISBI-2012 数据副本：

- Kaggle 数据集：  
  https://www.kaggle.com/datasets/hamzamohiuddin/isbi-2012-challenge

当前产物：

- 数据：`data/isbi2012_raw/`
- 训练审计：`data_summary/isbi2012_train/`
- 测试审计：`data_summary/isbi2012_test/`
- 训练实验：`exp_202_isbi2012_baseline_adamw_amp`
- 本地 test Dice：`0.9104232788`

更严格的“原论文级”复刻仍需要注意：

1. U-Net 原论文主要报告 ISBI Cell Tracking Challenge 2015 的 PhC-U373 和 DIC-HeLa 等结果，以及 EM segmentation 示例；当前 Kaggle 副本是 ISBI-2012 EM segmentation 数据，不等于完整论文挑战协议。
2. 原论文作者提供 Caffe/Matlab release：  
   https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
3. Cell Tracking Challenge 数据通常需要注册：  
   https://celltrackingchallenge.net/2d-datasets/
4. 若后续目标是严格复刻论文 challenge 指标，需要下载对应 CTC 数据集、遵循 challenge 的训练/测试协议，并使用官方 evaluation 工具，而不是只用当前仓库的 Dice。
