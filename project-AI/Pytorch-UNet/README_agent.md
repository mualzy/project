# Agent Reproduction README

This directory contains the original `milesial/Pytorch-UNet` repository plus local experiment assets for reproducible execution on a Windows SSH laptop.

## Current Runtime

```powershell
$env:PATH='D:\project-AI\Pytorch-UNet\.venv_cuda121\Scripts;C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;C:\Program Files\NVIDIA Corporation\NVSMI'
$env:WANDB_MODE='disabled'
.\.venv_cuda121\Scripts\python.exe train.py --epochs 2 --batch-size 4 --scale 0.125 --validation 10 --learning-rate 1e-4 --amp --num-workers 0 --optimizer adamw --save-best --abort-on-nonfinite --checkpoint-dir checkpoints\exp_110_full_adamw_amp --img-dir data\carvana_real\imgs --mask-dir data\carvana_real\masks
```

## Notes

- CUDA runtime is available in `.venv_cuda121` with `torch 2.5.1+cu121`.
- Real Carvana data is present at `data/carvana_real`; audited result: 5088 valid image/mask pairs, zero anomalies.
- Best full-data run is `exp_110_full_adamw_amp`, max/final validation Dice 0.9913 on all 5088 real pairs.
- Full 5088-pair RMSprop AMP attempts `exp_102` and `exp_105` are recorded as failed diagnostics because epoch 2 produced `nan` losses.
