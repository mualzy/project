# Smoke Test Summary

- Experiment: exp_001_smoke
- Time: 2026-04-30
- Purpose: validate data loading, model forward, loss, backward pass, checkpoint save, and prediction script.
- Dataset: 48 synthetic Carvana-compatible samples, 128x128, binary masks.
- Command: `python train.py --epochs 1 --batch-size 2 --scale 0.5 --validation 20 --learning-rate 1e-4 --num-workers 0 --checkpoint-dir checkpoints/exp_001`
- Device: CPU-only PyTorch runtime. GPU unavailable in PyTorch because CUDA wheel install timed out.
- Runtime: 40.12 seconds.
- Checkpoint: `checkpoints/exp_001/checkpoint_epoch1.pth`
- Prediction command: `python predict.py --model checkpoints/exp_001/checkpoint_epoch1.pth --input ... --output ... --scale 0.5`
- Prediction outputs: `results/predictions/exp_001/`
- Figures:
  - `results/figures/training_curves/exp_001_loss_curve.png`
  - `results/figures/training_curves/exp_001_metric_curve.png`
  - `results/figures/prediction_samples/exp_001_pred_sample_*.png`
- Status: success.
- Notes: This validates engineering chain only; it is not a real Carvana benchmark due missing Kaggle data.
