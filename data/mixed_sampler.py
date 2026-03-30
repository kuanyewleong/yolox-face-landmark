import random
from torch.utils.data import Sampler


class RatioConcatSampler(Sampler):
    def __init__(self, concat_dataset, num_samples, wider_ratio=0.5, seed=0):
        self.dataset = concat_dataset
        self.num_samples = num_samples
        self.wider_len = len(concat_dataset.datasets[0])
        self.wflw_len = len(concat_dataset.datasets[1])
        self.wider_ratio = wider_ratio
        self.seed = seed

    def __iter__(self):
        g = random.Random(self.seed)
        for _ in range(self.num_samples):
            if g.random() < self.wider_ratio:
                yield g.randrange(self.wider_len)
            else:
                yield self.wider_len + g.randrange(self.wflw_len)

    def __len__(self):
        return self.num_samples