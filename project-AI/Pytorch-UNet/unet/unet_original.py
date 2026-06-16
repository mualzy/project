"""Original valid-convolution U-Net variant from the 2015 paper."""

import torch
import torch.nn as nn


def center_crop(feature: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Crop BCHW tensor to target height/width around its center."""
    _, _, h, w = feature.shape
    target_h, target_w = target_hw
    if target_h > h or target_w > w:
        raise ValueError(f"Cannot crop tensor of shape {(h, w)} to {(target_h, target_w)}")
    top = (h - target_h) // 2
    left = (w - target_w) // 2
    return feature[:, :, top:top + target_h, left:left + target_w]


class OriginalDoubleConv(nn.Module):
    """Two unpadded 3x3 convolutions followed by ReLU, without BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class OriginalDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = OriginalDoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.maxpool(x))


class OriginalUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = OriginalDoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = center_crop(skip, (x.shape[-2], x.shape[-1]))
        return self.conv(torch.cat([skip, x], dim=1))


class OriginalUNet(nn.Module):
    """Caffe-style U-Net: valid convolutions, cropped skips, transposed up-convs."""

    def __init__(self, n_channels: int, n_classes: int, base_channels: int = 64):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = False
        self.architecture = "original"
        self._use_checkpointing = False

        c = base_channels
        self.inc = OriginalDoubleConv(n_channels, c)
        self.down1 = OriginalDown(c, c * 2)
        self.down2 = OriginalDown(c * 2, c * 4)
        self.down3 = OriginalDown(c * 4, c * 8)
        self.down4 = OriginalDown(c * 8, c * 16)
        self.up1 = OriginalUp(c * 16, c * 8)
        self.up2 = OriginalUp(c * 8, c * 4)
        self.up3 = OriginalUp(c * 4, c * 2)
        self.up4 = OriginalUp(c * 2, c)
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    def _run(self, module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        if self._use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self._run(self.inc, x)
        x2 = self._run(self.down1, x1)
        x3 = self._run(self.down2, x2)
        x4 = self._run(self.down3, x3)
        x5 = self._run(self.down4, x4)
        x = self._run(self.up1, x5, x4)
        x = self._run(self.up2, x, x3)
        x = self._run(self.up3, x, x2)
        x = self._run(self.up4, x, x1)
        return self.outc(x)

    def use_checkpointing(self) -> None:
        self._use_checkpointing = True
