import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_ctc_weight_maps import make_weight_map_exact
from unet import build_unet
from unet.unet_original import center_crop
from utils.overlap_tile import overlap_tile_logits


def check_shape_and_layers() -> None:
    model = build_unet("original", n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 572, 572)
    y = model(x)
    assert tuple(y.shape) == (1, 2, 388, 388), tuple(y.shape)
    conv3 = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3)]
    assert len(conv3) == 18, len(conv3)
    assert all(m.padding == (0, 0) for m in conv3)
    assert sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)) == 0
    conv_or_up = sum(
        1 for m in model.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))
    )
    assert conv_or_up == 23, conv_or_up


def check_crop_alignment() -> None:
    model = build_unet("original", n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 572, 572)
    x1 = model.inc(x)
    x2 = model.down1(x1)
    x3 = model.down2(x2)
    x4 = model.down3(x3)
    x5 = model.down4(x4)
    u1 = model.up1.up(x5)
    u2 = model.up2.up(model.up1.conv(torch.cat([center_crop(x4, u1.shape[-2:]), u1], dim=1)))
    u3 = model.up3.up(model.up2.conv(torch.cat([center_crop(x3, u2.shape[-2:]), u2], dim=1)))
    u4 = model.up4.up(model.up3.conv(torch.cat([center_crop(x2, u3.shape[-2:]), u3], dim=1)))
    assert x1.shape[-2:] == (568, 568)
    assert x2.shape[-2:] == (280, 280)
    assert x3.shape[-2:] == (136, 136)
    assert x4.shape[-2:] == (64, 64)
    assert u1.shape[-2:] == (56, 56)
    assert u2.shape[-2:] == (104, 104)
    assert u3.shape[-2:] == (200, 200)
    assert u4.shape[-2:] == (392, 392)


def check_paper_loss() -> None:
    logits = torch.tensor([[[[2.0, 0.0], [0.0, 2.0]], [[0.0, 2.0], [2.0, 0.0]]]])
    target = torch.tensor([[[0, 1], [1, 0]]])
    weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    pixel_loss = F.cross_entropy(logits, target, reduction="none")
    loss = (pixel_loss * weights).mean()
    manual = -F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
    assert torch.allclose(pixel_loss, manual)
    assert torch.allclose(loss, (manual * weights).mean())


def check_weight_maps() -> None:
    label = np.zeros((64, 64), dtype=np.int32)
    label[18:44, 8:28] = 1
    label[18:44, 36:56] = 2
    weight, stats = make_weight_map_exact(
        label,
        w0=10.0,
        sigma=5.0,
        class_weight_bg=1.0,
        class_weight_fg=1.0,
        class_balance="manual",
    )
    assert stats["objects"] == 2
    assert float(weight[31, 32]) > float(weight[3, 3])
    single = np.zeros((64, 64), dtype=np.int32)
    single[18:44, 18:44] = 1
    single_weight, single_stats = make_weight_map_exact(
        single,
        w0=10.0,
        sigma=5.0,
        class_weight_bg=1.0,
        class_weight_fg=1.0,
        class_balance="manual",
    )
    assert single_stats["objects"] == 1
    assert np.allclose(single_weight, 1.0)


def check_overlap_tile() -> None:
    torch.manual_seed(0)
    model = build_unet("original", n_channels=1, n_classes=2).eval()
    image = torch.randn(1, 388, 388)
    margin = 92
    padded = F.pad(image.unsqueeze(0), [margin, margin, margin, margin], mode="reflect")
    full_logits = model(padded)
    tiled_logits = overlap_tile_logits(model, image, device=torch.device("cpu"), tile_size=572)
    assert torch.allclose(full_logits, tiled_logits, atol=1e-5), (full_logits - tiled_logits).abs().max()


def main() -> None:
    check_shape_and_layers()
    check_crop_alignment()
    check_paper_loss()
    check_weight_maps()
    check_overlap_tile()
    print("original_method_checks=passed")


if __name__ == "__main__":
    main()
