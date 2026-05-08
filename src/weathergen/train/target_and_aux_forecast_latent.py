# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch

from weathergen.train.target_and_aux_module_base import TargetAndAuxModuleBase, TargetAuxOutput


class ForecastLatentTarget(TargetAndAuxModuleBase):
    """
    JEPA-style encoder targets for FE latent alignment.

    Runs the (frozen-gradient) encoder on ground-truth future samples and
    stores the resulting latent vectors so the FE alignment loss can minimise
        L = ||z_FE - stop_grad(z_enc)||²

    The compute() method receives the TARGET BatchSamples (future ground-truth),
    encodes them once, and stores the result for the first output step.
    For multi-step validation the target is only available at step 0; the loss
    module skips steps where 'encoder_z' is absent.
    """

    def __init__(self, cf, model, **kwargs):
        pass

    def compute(self, bidx, batch, model_params, model, **kwargs) -> TargetAuxOutput:
        output_idxs = batch.get_output_idxs()
        targets = TargetAuxOutput(batch.get_output_len(), output_idxs)

        if not output_idxs:
            return targets

        with torch.no_grad():
            # batch is the target BatchSamples whose source_tokens_cells hold
            # the ground-truth future state (e.g. T+1 for 1-step forecasting).
            tokens, _ = model.forward_latent(model_params, batch)
            # tokens: (B, num_tokens, d) — encoder output for future ground-truth
            targets.add_latent_target(output_idxs[0], "encoder_z", tokens.detach())

        return targets

    def to_device(self, device) -> "ForecastLatentTarget":
        return self
