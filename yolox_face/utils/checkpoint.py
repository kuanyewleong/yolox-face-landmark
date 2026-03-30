from pathlib import Path
import torch


def save_checkpoint(state, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, path: str, optimizer=None, scaler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt