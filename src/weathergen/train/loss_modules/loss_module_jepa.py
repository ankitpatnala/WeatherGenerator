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
from omegaconf import DictConfig

from weathergen.train.loss_modules.loss_module_base import LossModuleBase, LossValues
from weathergen.utils.train_logger import Stage

_logger = logging.getLogger(__name__)


class LossLatentJEPA(LossModuleBase):
    """
    JEPA-style alignment loss: MSE between FE patch tokens at the last forecast
    step and stop-gradient encoder tokens at the same timestep.

    The MSE is computed inside model.forward() to avoid holding a full
    [B, 12288, D] tensor in the output dict. This module just reads the
    precomputed scalar from output.latent[last_step]["latent_alignment_loss"]
    and scales it by the configured weight.
    """

    def __init__(self, cf: DictConfig, mode_cfg: DictConfig, stage: Stage, device: str, **loss_fcts):
        LossModuleBase.__init__(self)
        self.cf = cf
        self.stage = stage
        self.device = device
        self.name = "LossLatentJEPA"

        params = next(iter(loss_fcts.values()), {}) if loss_fcts else {}
        self.weight = float(params.get("weight", 1.0))

    def compute_loss(self, preds, targets, metadata, **kwargs) -> LossValues:
        loss = torch.tensor(0.0, device=self.device, requires_grad=False)

        for step_pred in preds.latent:
            alignment_loss = step_pred.get("latent_alignment_loss", None)
            if alignment_loss is not None:
                loss = alignment_loss * self.weight
                break  # only one step carries this value (the last forecast step)

        return LossValues(
            loss=loss,
            losses_all={"jepa_alignment": loss.detach().item()},
            stddev_all={},
        )
