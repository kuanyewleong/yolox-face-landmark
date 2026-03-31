import torch


@torch.no_grad()
def evaluate_simple(model, data_loader, device, loss_fn):
    model.eval()
    loss_meter = {"loss": 0.0, "loss_iou": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_lmk": 0.0}
    n = 0

    for i, batch in enumerate(data_loader):
        batch["images"] = batch["images"].to(device, non_blocking=True)
        outputs = model(batch["images"])
        losses = loss_fn(outputs, batch)

        bs = batch["images"].shape[0]
        for k in loss_meter:
            loss_meter[k] += losses[k].item() * bs
        n += bs

        if i % 20 == 0 or i == len(data_loader) - 1:
            print(f"[val] iter {i + 1}/{len(data_loader)}", flush=True)

    for k in loss_meter:
        loss_meter[k] /= max(n, 1)
    return loss_meter