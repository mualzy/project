# Experiment Notes

- `exp_001_smoke`: minimal synthetic dataset link verification.
- `exp_002_baseline_cpu_synth`: CPU baseline on synthetic Carvana-compatible data because Kaggle credentials and CUDA wheel installation are blocked.
- `exp_003_lr_high_cpu_synth`: learning-rate extension, lr=5e-4, 2 epochs. Max validation Dice 0.9430.
- `exp_004_scale1_cpu_synth`: input-scale extension, scale=1.0, 2 epochs. Max validation Dice 0.8626.

## Interpretation

- `exp_002` is the best current result and provides the cleanest report material.
- `exp_003` converged quickly with a higher learning rate but did not exceed the 3-epoch baseline.
- `exp_004` used higher resolution but was slower on CPU and needs more training or GPU to be useful.

## Real Carvana GPU Experiments

- `exp_101_real_smoke_clean`: 64 real pairs, CUDA AMP, batch size 2, scale 0.125, 1 epoch. Purpose was chain validation after reboot. It completed training, checkpointing, prediction, and panel generation.
- `exp_102_real_baseline_amp`: full 5088-pair AMP attempt, lr=1e-4, batch size 4, scale 0.125, 2 epochs. It ran but failed numerically with `nan` losses in epoch 2.
- `exp_105_real_baseline_amp_lr1e5`: full 5088-pair AMP attempt, lr=1e-5. Epoch 1 reached max validation Dice 0.9742, but epoch 2 still collapsed to `nan`, so this is not accepted as a stable baseline.
- `exp_106_real512_noamp_lr1e5`: 512-pair controlled baseline without AMP. Stable, max validation Dice 0.8529.
- `exp_107_real512_amp_lr1e5`: 512-pair AMP extension. Stable, fastest among the 512-pair 2-epoch runs, max validation Dice 0.9443. This is the best stable real-data result.
- `exp_108_real512_scale025_amp`: 512-pair scale 0.25 extension, batch size 2, 1 epoch. Stable, max validation Dice 0.8980, but slower per epoch than scale 0.125.

## Real-Data Interpretation

- The real Carvana data is downloaded, audited, and usable: 5088 valid image/mask pairs with zero pairing or decoding anomalies.
- The recommended current recipe on this Windows laptop is CUDA + AMP, `--batch-size 4`, `--scale 0.125`, `--learning-rate 1e-5`, `--num-workers 0`, first validated on a subset before full-dataset use.
- Full-dataset 2-epoch training is runnable but not yet stable with the tested recipes. Any next full run should add non-finite loss detection, best checkpoint saving, and probably a safer optimizer or learning-rate schedule.

## Full-Dataset Repair Experiments

- `exp_109_full_resume_amp_lr1e6`: attempted to resume from `exp_105` epoch1 with lr=1e-6 and AMP. It failed at step 5 with non-finite loss. This confirmed that the earlier high-Dice RMSprop checkpoint was not a safe continuation point.
- `exp_110_full_adamw_amp`: full 5088-pair run from scratch with AdamW, AMP, batch size 4, scale 0.125, lr=1e-4, 2 epochs, `--save-best`, and `--abort-on-nonfinite`. It completed successfully with max/final validation Dice 0.9913.

## Updated Recommendation

- Use `exp_110_full_adamw_amp` as the accepted full-data reproduction baseline.
- Keep RMSprop full-run results as diagnostic failures. On this Windows/RTX 4070 Laptop setup, RMSprop with momentum 0.999 produced extreme weight growth and eventual `nan`; AdamW fixed the failure while preserving AMP speed.
