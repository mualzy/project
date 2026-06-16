# Project AI Segmentation Experiments

This repository bundles two segmentation code projects in one GitHub repository:

- `Pytorch-UNet/`: a U-Net based segmentation project with training, prediction, evaluation, and experiment utility scripts.
- `Fast-SCNN-pytorch/`: a Fast-SCNN based segmentation project with training, evaluation, and model/data-loader code.

Only source code, lightweight notes, and setup files are included. Local virtual environments, datasets, checkpoints, logs, generated results, model weights, papers, and media files are intentionally excluded.

## Repository Layout

```text
.
|-- Pytorch-UNet/
|   |-- train.py
|   |-- predict.py
|   |-- evaluate.py
|   |-- unet/
|   |-- utils/
|   `-- scripts/
`-- Fast-SCNN-pytorch/
    |-- train.py
    |-- eval.py
    |-- demo.py
    |-- models/
    |-- data_loader/
    `-- utils/
```

## Basic Setup

Use a separate Python environment for each project.

For `Pytorch-UNet`:

```bash
cd Pytorch-UNet
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For `Fast-SCNN-pytorch`, install PyTorch for your CUDA/CPU environment first, then install the project's Python dependencies as needed by its README and scripts.

Datasets and trained weights are not committed. Put them in each project's expected local data/checkpoint directories before running training or evaluation.
