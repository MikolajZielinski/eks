"""Scheduler Classes"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler


@dataclass
class ChainedSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: ChainedScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    gamma: float = 0.33
    """The learning rate decay factor."""
    milestones: Tuple[float, ...] = (0.5, 0.75, 0.9)
    """The milestone steps at which to decay the learning rate."""


class ChainedScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: ChainedSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.config.milestones,
            gamma=self.config.gamma,
        )

        scheduler = lr_scheduler.ChainedScheduler(
            [
                lr_scheduler.LinearLR(optimizer=optimizer, 
                                      start_factor=0.01, 
                                      total_iters=100),
                lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                         milestones=[int(m * self.config.max_steps) for m in self.config.milestones], 
                                         gamma=self.config.gamma),
            ]
        )

        return scheduler