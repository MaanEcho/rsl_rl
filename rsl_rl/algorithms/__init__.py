# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different learning algorithms."""

from .distillation import Distillation
from .ppo import PPO

from .DreamWaQ.ppo_DreamWaQ import PPODreamWaQ
from .DreamWaQ.ppo_DreamWaQV1 import PPODreamWaQV1
from .HIMLoco.him_ppo import HIMPPO
from .VQVAE.ppo_VQVAE import PPOVQVAE

__all__ = ["PPO", "Distillation", "PPODreamWaQ", "PPODreamWaQV1", "HIMPPO", "PPOVQVAE"]
