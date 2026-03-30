import copy
import math
import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.9998):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)