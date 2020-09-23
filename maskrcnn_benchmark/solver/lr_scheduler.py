# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch
import math
pi = 3.141592657


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        # self.decay_step = decay_step
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


    # def get_lr(self):
    #     warmup_factor = 1
    #     if self.last_epoch < self.warmup_iters:
    #         if self.warmup_method == "constant":
    #             warmup_factor = self.warmup_factor
    #         elif self.warmup_method == "linear":
    #             alpha = float(self.last_epoch) / self.warmup_iters
    #             warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        
    #     global_step = min(self.last_epoch, self.decay_step)
    #     cosine_decay = 0.5*(1+math.cos(pi*global_step/self.decay_step))
    #     decayed = (1-self.cos_alpha)*cosine_decay+self.cos_alpha
    #     # print([warmup_factor, decayed, self.decay_step, self.last_epoch])
    #     return [base_lr * warmup_factor * decayed for base_lr in self.base_lrs]