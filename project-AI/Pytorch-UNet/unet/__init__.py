from .unet_model import UNet
from .unet_original import OriginalUNet


def build_unet(architecture: str, n_channels: int, n_classes: int, bilinear: bool = False):
    if architecture == "milesial":
        return UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    if architecture == "original":
        if bilinear:
            raise ValueError("The original valid-conv U-Net uses transposed convolutions, not bilinear upsampling")
        return OriginalUNet(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(f"Unknown architecture: {architecture}")
