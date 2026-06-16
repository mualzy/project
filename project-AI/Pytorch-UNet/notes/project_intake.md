# Project Intake

- Repository: milesial/Pytorch-UNet
- Local path: D:\project-AI\Pytorch-UNet
- Commit: 21d7850f2af30a9695bbeea75f3136aa538cfc4a
- Paper target: U-Net: Convolutional Networks for Biomedical Image Segmentation
- Implementation target: PyTorch U-Net semantic segmentation for Carvana-style binary masks.

## Core Files

- `train.py`: CLI training entry. Builds `UNet`, loads `data/imgs` and `data/masks`, splits train/validation, trains with RMSprop, CE/BCE plus Dice loss, evaluates Dice, and saves checkpoints.
- `predict.py`: CLI inference entry. Loads a `.pth` checkpoint, runs image preprocessing and model inference, saves predicted mask images.
- `utils/data_loading.py`: `BasicDataset` and `CarvanaDataset`. Carvana masks are matched as `<image_id>_mask.*`.
- `unet/unet_model.py` and `unet/unet_parts.py`: U-Net encoder/decoder definition.
- `utils/dice_score.py`: Dice coefficient and Dice loss.
- `evaluate.py`: validation loop and Dice metric.
- `scripts/download_data.sh` / `scripts/download_data.bat`: Kaggle Carvana data download helpers.

## Default Workflow

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
bash scripts/download_data.sh
.\.venv\Scripts\python.exe train.py --amp
.\.venv\Scripts\python.exe predict.py -i image.jpg -o output.jpg --model MODEL.pth
```

## Data Requirements

- Images must be placed directly in `data/imgs`.
- Masks must be placed directly in `data/masks`.
- Carvana mask names use `_mask` suffix, for example `id.jpg` and `id_mask.gif`.
- The loader is greedy and does not accept subdirectories.

## Key Risks

- Original Kaggle data requires user Kaggle credentials and competition access.
- Windows Bash availability is not guaranteed; the repo includes a `.bat` downloader but credentials are still required.
- Original `train.py` used `num_workers=os.cpu_count()`, which is aggressive on a 16 GB Windows laptop.
- Global Python pip/torch installation on this machine is broken; isolated `.venv` is required.
- CUDA PyTorch wheel download from the PyTorch CUDA index timed out repeatedly; current environment is CPU-only until CUDA wheel installation succeeds.
- W&B online login should not block unattended SSH runs; use `WANDB_MODE=offline` or `WANDB_MODE=disabled`.

## Local Engineering Adjustments

- Added `--num-workers` and `--checkpoint-dir` to `train.py` so experiments can use stable Windows DataLoader settings and isolated checkpoint directories.
- Added local scripts for synthetic Carvana-compatible data, dataset audit, prediction panels, and training curve extraction.
