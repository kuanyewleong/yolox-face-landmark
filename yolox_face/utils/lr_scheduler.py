import math


class LRScheduler:
    def __init__(self, optimizer, base_lr, min_lr_ratio, total_iters, warmup_iters):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = base_lr * min_lr_ratio
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.it = 0

    def step(self):
        self.it += 1
        if self.it <= self.warmup_iters:
            lr = self.base_lr * self.it / max(1, self.warmup_iters)
        else:
            t = (self.it - self.warmup_iters) / max(1, self.total_iters - self.warmup_iters)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr