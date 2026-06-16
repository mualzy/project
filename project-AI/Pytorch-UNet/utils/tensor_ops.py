import torch


def center_crop_tensor(tensor: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Center-crop a BCHW or BHW tensor to target height/width."""
    target_h, target_w = target_hw
    if tensor.ndim == 4:
        h, w = tensor.shape[-2:]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return tensor[:, :, top:top + target_h, left:left + target_w]
    if tensor.ndim == 3:
        h, w = tensor.shape[-2:]
        top = (h - target_h) // 2
        left = (w - target_w) // 2
        return tensor[:, top:top + target_h, left:left + target_w]
    raise ValueError(f"Expected BCHW or BHW tensor, got shape {tuple(tensor.shape)}")


def crop_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Center-crop tensor spatial dims to match reference when needed."""
    target_hw = reference.shape[-2:]
    if tensor.shape[-2:] == target_hw:
        return tensor
    return center_crop_tensor(tensor, target_hw)
