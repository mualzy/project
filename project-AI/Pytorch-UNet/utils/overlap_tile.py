import torch
import torch.nn.functional as F


def _starts(length: int, window: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, window))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.inference_mode()
def infer_valid_output_hw(
    model: torch.nn.Module,
    tile_hw: tuple[int, int],
    device: torch.device,
    amp: bool = False,
) -> tuple[int, int]:
    tile_h, tile_w = tile_hw
    dummy = torch.zeros((1, model.n_channels, tile_h, tile_w), device=device, dtype=torch.float32)
    dummy = dummy.to(memory_format=torch.channels_last)
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        out = model(dummy)
    return out.shape[-2], out.shape[-1]


@torch.inference_mode()
def overlap_tile_logits(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    tile_size: int = 572,
    amp: bool = False,
) -> torch.Tensor:
    """Return full-size logits for CHW image_tensor using valid-output tiles."""
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape {tuple(image_tensor.shape)}")

    model.eval()
    image = image_tensor.unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
        memory_format=torch.channels_last,
    )
    _, _, h, w = image.shape
    tile_hw = (tile_size, tile_size)
    out_h, out_w = infer_valid_output_hw(model, tile_hw, device, amp=amp)
    margin_h = (tile_size - out_h) // 2
    margin_w = (tile_size - out_w) // 2
    if margin_h < 0 or margin_w < 0:
        raise ValueError(f"Tile size {tile_size} produced larger output {(out_h, out_w)}")

    pad_mode = "reflect" if h > margin_h and w > margin_w else "replicate"
    padded = F.pad(image, [margin_w, margin_w, margin_h, margin_h], mode=pad_mode)

    logits = torch.zeros((1, model.n_classes, h, w), device=device, dtype=torch.float32)
    counts = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)
    y_starts = _starts(h, out_h)
    x_starts = _starts(w, out_w)

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for y in y_starts:
            for x in x_starts:
                tile = padded[:, :, y:y + tile_size, x:x + tile_size]
                tile_logits = model(tile).float()
                take_h = min(out_h, h - y)
                take_w = min(out_w, w - x)
                logits[:, :, y:y + take_h, x:x + take_w] += tile_logits[:, :, :take_h, :take_w]
                counts[:, :, y:y + take_h, x:x + take_w] += 1

    return logits / counts.clamp_min(1)
