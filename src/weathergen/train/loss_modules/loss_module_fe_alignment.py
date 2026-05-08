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


class LossLatentFEAlignment(LossModuleBase):
    """
    JEPA-style FE latent alignment loss.

    Minimises  L = ||z_FE_t - stop_grad(z_enc_t)||²  where z_enc_t is the
    frozen-encoder output for the ground-truth future state.  This keeps FE
    latents on the encoder manifold in both direction and magnitude.

    Config keys (under loss_fcts):
        only_patch_tokens : bool,  default True  — exclude register/class tokens
        normalize         : bool,  default False — L2-normalise before MSE
                                                   (turns it into cosine alignment)
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
        self.name = "LossLatentFEAlignment"
        params = next(iter(loss_fcts.values()), {})
        self.only_patch_tokens = params.get("only_patch_tokens", True)
        self.normalize = params.get("normalize", False)
        self.num_aux_tokens = cf.num_register_tokens + cf.num_class_tokens

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        losses_all: dict = {}

        if targets is None:
            losses_all["fe_align_mse"] = 0.0
            return LossValues(loss=loss, losses_all=losses_all, stddev_all={})

        count = 0
        acc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for pred_dict, target_dict in zip(preds.latent, targets.latent):
            if not pred_dict or not target_dict:
                continue
            if "latent_state" not in pred_dict or "encoder_z" not in target_dict:
                continue

            z_fe = pred_dict["latent_state"].z_pre_norm   # (B, T, d)
            z_enc = target_dict["encoder_z"]              # (B, T, d) — already detached

            if self.only_patch_tokens and self.num_aux_tokens > 0 and z_fe.dim() == 3:
                z_fe = z_fe[:, self.num_aux_tokens:, :]
                z_enc = z_enc[:, self.num_aux_tokens:, :]

            if self.normalize:
                z_fe = F.normalize(z_fe, dim=-1)
                z_enc = F.normalize(z_enc, dim=-1)

            step_loss = F.mse_loss(z_fe, z_enc.detach())
            acc_loss = acc_loss + step_loss
            count += 1

        if count > 0:
            loss = acc_loss / count

        losses_all["fe_align_mse"] = loss.detach().item()
        return LossValues(loss=loss, losses_all=losses_all, stddev_all={})
