import torch
from bisect import bisect_right
from configs import *


def make_optimizer(model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = SOLVER_BASE_LR
        weight_decay = SOLVER_WEIGHT_DECAY
        if "bias" in key:
            lr = SOLVER_BASE_LR * SOLVER_BIAS_LR_FACTOR
            weight_decay = SOLVER_WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if SOLVER_OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, SOLVER_OPTIMIZER_NAME)(params, momentum= SOLVER_MOMENTUM)
    else:
        optimizer = getattr(torch.optim, SOLVER_OPTIMIZER_NAME)(params)
    return optimizer

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
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size=2,
        gamma=0.9,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        # if not list(milestones) == sorted(milestones):
        #     raise ValueError(
        #         "Milestones should be a list of" " increasing integers. Got {}",
        #         milestones,
        #     )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]