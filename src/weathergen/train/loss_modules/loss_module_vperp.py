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

_logger = logging.getLogger(__name__)


class LossLatentVPerpNorm(LossModuleBase):
    """
    Penalises tokens where the FE output is parallel to prev_tokens (v_perp_norm ≈ 0).

    When fe_enforce_cosine=True, model.forward() geometrically projects each FE
    step to a fixed cosine angle. The projection uses v_perp — the component of
    the FE output orthogonal to prev_tokens — as the rotation direction. If
    v_perp ≈ 0 the FE is not providing a useful rotation direction and the
    enforcement has nothing to work with.

    Hinge loss: relu(v_perp_min - v_perp_norm)^2 per token, averaged over tokens
    and steps. Zero when v_perp_norm >= v_perp_min, quadratic below.

    v_perp_norm is stored in output.latent[step]["v_perp_norm"] by model.forward()
    only when fe_enforce_cosine=True — this loss is a no-op otherwise.
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **loss_fcts):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentVPerpNorm"

        params = next(iter(loss_fcts.values()), {}) if loss_fcts else {}
        self.v_perp_min = float(params.get("v_perp_min", 0.1))

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        acc_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        count = 0

        for step_pred in preds.latent:
            v_perp_norm = step_pred.get("v_perp_norm", None)
            if v_perp_norm is None:
                continue
            step_loss = F.relu(self.v_perp_min - v_perp_norm).pow(2).mean()
            acc_loss = acc_loss + step_loss
            count += 1

        loss = acc_loss / count if count > 0 else acc_loss
        return LossValues(loss=loss, losses_all={"v_perp_norm": loss.detach().item()}, stddev_all={})
