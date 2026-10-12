from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import DenseNet121


def build_model() -> torch.nn.Module:
    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)


def load_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint: Any = torch.load(path, map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = build_model().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def probability(model: torch.nn.Module, tensor: torch.Tensor, device: torch.device) -> float:
    logits = model(tensor.to(device=device, dtype=torch.float32))
    return float(torch.softmax(logits, dim=1)[0, 1].detach().cpu())
