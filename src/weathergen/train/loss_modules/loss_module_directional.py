# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

logger = logging.getLogger(__name__)


class LossDirectionalMatching(LossModuleBase):
    """
    Directional anti-alignment loss for autoregressive latent rollout.

    Penalizes high cosine similarity between consecutive forecast latent states,
    directly countering the directional freeze observed in the latent magnitude
    and directional drift plots.  No teacher model required — purely self-supervised.

    The loss is a hinge on cosine similarity:
        L = mean( relu( cos_sim(z_t, z_{t+1}) - threshold ) )

    A threshold < 1.0 means the loss only activates when consecutive states are
    more aligned than the threshold, leaving the model free to be conservative
    when the atmospheric transition genuinely calls for it.

    Config keys (under loss_fcts):
        threshold          : float, default 0.8 — hinge threshold on cosine similarity
        only_patch_tokens  : bool,  default True — ignore register/class tokens
    """

    def __init__(
        self,
        cf: DictConfig,
        mode_cfg: DictConfig,
        stage: Stage,
        device: str,
        **loss_fcts,
    ):
        super().__init__()
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossDirectionalMatching"
        # loss_fcts follows the same convention as LossPhysical:
        # a dict of named param-dicts, e.g. {"params": {"threshold": 0.8, ...}}
        params = next(iter(loss_fcts.values()), {})
        self.threshold = params.get("threshold", 0.8)
        self.only_patch_tokens = params.get("only_patch_tokens", True)
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses_all: dict = {}

        # Cosine similarities were computed inline during rollout (scalar per step pair).
        # This avoids storing full token tensors across steps.
        cos_sim_scalars = [
            d["dir_cos_sim"]
            for d in preds.latent
            if d is not None and "dir_cos_sim" in d
        ]

        if not cos_sim_scalars:
            losses_all["dir_cos_sim"] = 0.0
            return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

        stacked = torch.stack(cos_sim_scalars)
        if self.threshold > 0.0:
            # hinge: only penalise above threshold
            loss = F.relu(stacked - self.threshold).mean()
        else:
            # threshold=0: penalise raw cosine similarity directly (gradient everywhere)
            loss = stacked.mean()
        losses_all["dir_cos_sim"] = stacked.detach().mean().item()
        losses_all["dir_hinge"] = loss.detach().item()

        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})
