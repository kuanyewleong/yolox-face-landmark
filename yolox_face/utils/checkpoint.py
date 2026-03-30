from pathlib import Path
import torch


def save_checkpoint(state, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, path: str, optimizer=None, scaler=None, map_location="cpu", strict=False):
    ckpt = torch.load(path, map_location=map_location)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=strict)

    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and isinstance(ckpt, dict) and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def save_final_model(model, cfg, path: str):
    state_dict = model.state_dict()
    payload = {
        "model": state_dict,
        "cfg": cfg,
    }
    save_checkpoint(payload, path)