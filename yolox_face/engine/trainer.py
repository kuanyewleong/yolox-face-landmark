import json
import os
import time
import torch
from torch.amp import GradScaler, autocast
from yolox_face.engine.evaluator import evaluate_simple
from yolox_face.losses.yolox_loss import YOLOXLoss
from yolox_face.utils.checkpoint import load_checkpoint, save_checkpoint
from yolox_face.utils.ema import ModelEMA
from yolox_face.utils.lr_scheduler import LRScheduler


def make_optimizer(model, cfg):
    train_cfg = cfg["train"]
    pg0, pg1, pg2 = [], [], []
    for _, m in model.named_modules():
        if hasattr(m, "bias") and isinstance(m.bias, torch.nn.Parameter):
            pg2.append(m.bias)
        if isinstance(m, torch.nn.BatchNorm2d) or "bn" in m.__class__.__name__.lower():
            if hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
                pg0.append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            pg1.append(m.weight)
    optimizer = torch.optim.SGD(pg0, lr=train_cfg["base_lr"], momentum=train_cfg["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": train_cfg["weight_decay"]})
    optimizer.add_param_group({"params": pg2})
    return optimizer

def train_one_epoch(model, ema, loader, optimizer, scheduler, scaler, loss_fn, device, epoch, total_epochs, amp, phase_name):
    model.train()
    meters = {"loss": 0.0, "loss_iou": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_lmk": 0.0}
    count = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        batch["images"] = images
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(amp and device.type == "cuda")):
            outputs = model(images)
            losses = loss_fn(outputs, batch)
            loss = losses["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update(model)

        bs = images.shape[0]
        for k in meters:
            meters[k] += losses[k].item() * bs
        count += bs

        if i % 20 == 0 or i == len(loader) - 1:
            print(
                f"[{phase_name}] epoch {epoch + 1}/{total_epochs} iter {i + 1}/{len(loader)} "
                f"loss={losses['loss'].item():.4f} iou={losses['loss_iou'].item():.4f} "
                f"obj={losses['loss_obj'].item():.4f} cls={losses['loss_cls'].item():.4f} "
                f"lmk={losses['loss_lmk'].item():.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    for k in meters:
        meters[k] /= max(count, 1)
    meters["time"] = time.time() - t0
    return meters


def build_training_state(model, cfg, phase_name, checkpoint_path=None, resume=False):
    train_cfg = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = YOLOXLoss(
        num_classes=cfg["model"]["num_classes"],
        reg_weight=train_cfg["reg_weight"],
        obj_weight=train_cfg["obj_weight"],
        cls_weight=train_cfg["cls_weight"],
        lmk_weight=train_cfg["lmk_weight"],
    )
    optimizer = make_optimizer(model, cfg)
    use_amp = train_cfg["amp"] and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)
    total_epochs = train_cfg["epochs_phase1"] if phase_name == "phase1" else train_cfg["epochs_phase2"]
    ema = ModelEMA(model)

    start_epoch = 0
    best_loss = 1e9
    if checkpoint_path:
        ckpt = load_checkpoint(
            model,
            checkpoint_path,
            optimizer=optimizer if resume else None,
            scaler=scaler if resume else None,
            map_location="cpu",
            strict=False,
        )
        if isinstance(ckpt, dict):
            best_loss = float(ckpt.get("best_loss", best_loss))
            if resume:
                start_epoch = int(ckpt.get("epoch", 0))
                if "ema" in ckpt:
                    ema.ema.load_state_dict(ckpt["ema"], strict=False)

    total_iters = len_placeholder = None
    return {
        "model": model,
        "device": device,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "scaler": scaler,
        "ema": ema,
        "total_epochs": total_epochs,
        "start_epoch": start_epoch,
        "best_loss": best_loss,
    }

def run_phase(model, train_loader, val_loader, cfg, output_prefix, phase_name, checkpoint_path=None, resume=False):
    train_cfg = cfg["train"]
    output_dir = cfg["output_dir"]

    state = build_training_state(model, cfg, phase_name, checkpoint_path=checkpoint_path, resume=resume)
    model = state["model"]
    device = state["device"]
    loss_fn = state["loss_fn"]
    optimizer = state["optimizer"]
    scaler = state["scaler"]
    ema = state["ema"]
    total_epochs = state["total_epochs"]
    start_epoch = state["start_epoch"]
    best = state["best_loss"]

    total_iters = len(train_loader) * total_epochs
    warmup_iters = len(train_loader) * train_cfg["warmup_epochs"]
    scheduler = LRScheduler(optimizer, train_cfg["base_lr"], train_cfg["min_lr_ratio"], total_iters, warmup_iters)
    scheduler.it = start_epoch * len(train_loader)

    for epoch in range(start_epoch, total_epochs):
        train_stats = train_one_epoch(model, ema, train_loader, optimizer, scheduler, scaler, loss_fn, device, epoch, total_epochs, train_cfg["amp"], phase_name)
        print(f"[{phase_name}] train epoch={epoch + 1}: {json.dumps(train_stats, indent=2)}")

        if val_loader is not None and ((epoch + 1) % 5 == 0 or epoch + 1 == total_epochs):
            eval_model = ema.ema.to(device)
            val_stats = evaluate_simple(eval_model, val_loader, device, loss_fn)
            print(f"[{phase_name}] val epoch={epoch + 1}: {json.dumps(val_stats, indent=2)}")

            ckpt = {
                "model": eval_model.state_dict(),
                "ema": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_loss": min(best, val_stats["loss"]),
                "cfg": cfg,
                "phase": phase_name,
            }
            latest_path = os.path.join(output_dir, f"{output_prefix}_latest.pth")
            save_checkpoint(ckpt, latest_path)
            if val_stats["loss"] < best:
                best = val_stats["loss"]
                save_checkpoint(ckpt, os.path.join(output_dir, f"{output_prefix}_best.pth"))

    return ema.ema.cpu()