# Project AI

This public repository contains two semantic segmentation code projects prepared for coursework/research reproduction.

## Contents

- [`project-AI/Pytorch-UNet`](project-AI/Pytorch-UNet): U-Net segmentation project with training, prediction, evaluation, and experiment helper scripts.
- [`project-AI/Fast-SCNN-pytorch`](project-AI/Fast-SCNN-pytorch): Fast-SCNN segmentation project with model, data loader, training, evaluation, and demo code.
- [`project-AI/README.md`](project-AI/README.md): shared layout and setup notes for the two projects.

## What Is Included

The repository includes source code, lightweight notes, setup files, and project documentation.

The repository does not include local virtual environments, datasets, checkpoints, logs, generated results, model weights, papers, or media files. These files should be prepared locally before running training or evaluation.

## Basic Setup

Clone the repository:

```bash
git clone https://github.com/mualzy/project.git
cd project/project-AI
```

Create a separate Python environment for each project.

For `Pytorch-UNet`:

```bash
cd Pytorch-UNet
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

For `Fast-SCNN-pytorch`, create a virtual environment, install the PyTorch build that matches your CUDA/CPU environment, then follow the project README and scripts for training or evaluation.

## Data And Weights

Datasets and pretrained/trained weights are intentionally excluded from Git. Put them in the paths expected by each project, such as local `data/`, `datasets/`, `checkpoints/`, or `weights/` directories.

## Notes

The two projects are stored together under `project-AI/` so they can be shared through one public GitHub repository without mixing their code files.
