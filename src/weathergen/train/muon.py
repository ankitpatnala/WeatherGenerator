# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Muon optimizer — Momentum + Orthogonalization via Newton-Schulz.

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in neural networks"
           https://github.com/KellerJordan/modded-nanogpt

For 2D weight matrices the gradient is orthogonalized via Newton-Schulz iterations
before the update, ensuring no singular direction of the weight matrix dominates.
For 1D and scalar parameters (biases, LayerNorm scales) AdamW is used as fallback.

FSDP2 / DTensor compatibility
------------------------------
With FSDP2, weight parameters are DTensors (row-wise sharded across ranks).
Newton-Schulz requires the *full* matrix to compute the orthogonal factor.
We therefore all-gather the full gradient on every rank, orthogonalize it,
and re-shard the result before applying the update — ensuring every rank
applies the same geometrically-correct update to its local shard.
"""

import logging

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (quintic, 5 iterations default)
# ---------------------------------------------------------------------------

def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Approximate the orthogonal factor of G via Newton-Schulz iterations.

    Operates in bfloat16 for speed; always square-orients G so that
    the iteration converges (requires the larger dim on rows).

    Returns a matrix with the same shape as G whose singular values ≈ 1.
    """
    assert G.ndim == 2, "Newton-Schulz requires a 2D matrix"
    a, b, c = 3.4445, -4.7750, 2.0315   # quintic coefficients (Keller Jordan)

    X = G.to(torch.bfloat16) / (G.norm() + eps)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X

    return (X.T if transposed else X).to(G.dtype)


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """
    Muon: SGD with Nesterov momentum followed by Newton-Schulz orthogonalisation
    for 2D weight matrices.  1D / scalar parameters use AdamW.

    Parameters
    ----------
    muon_params :
        Iterable of 2D parameters to update with the Muon rule.
    lr : float
        Learning rate for the Muon update.
    momentum : float
        Nesterov momentum coefficient.
    ns_steps : int
        Number of Newton-Schulz iterations (5 is usually sufficient).
    adamw_params :
        Iterable of remaining (1D / scalar) parameters — updated with AdamW.
    adamw_lr : float
    adamw_betas : tuple
    adamw_eps : float
    adamw_wd : float
    """

    def __init__(
        self,
        muon_params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.1,
    ):
        muon_group = dict(
            params=list(muon_params),
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            use_muon=True,
        )
        groups = [muon_group]

        if adamw_params is not None:
            adamw_group = dict(
                params=list(adamw_params),
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_wd,
                use_muon=False,
            )
            groups.append(adamw_group)

        defaults = dict(use_muon=True)
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group):
        lr       = group["lr"]
        momentum = group["momentum"]
        ns_steps = group["ns_steps"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad

            # ── FSDP2 / DTensor: gather full gradient for Newton-Schulz ──
            is_dtensor = isinstance(grad, DTensor)
            if is_dtensor:
                full_grad = grad.full_tensor().float()
            else:
                full_grad = grad.float()

            assert full_grad.ndim == 2, (
                f"Muon param group expects 2D tensors, got shape {full_grad.shape}. "
                "Move non-2D params to the adamw_params group."
            )

            # ── Nesterov momentum ──
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(full_grad)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(full_grad)
            g_nesterov = full_grad.add(buf, alpha=momentum)

            # ── Newton-Schulz orthogonalisation ──
            ortho = _newton_schulz(g_nesterov, steps=ns_steps)
            # scale to match RMS of a Gaussian with same shape (keeps lr comparable to AdamW)
            ortho.mul_(max(1.0, full_grad.shape[0] / full_grad.shape[1]) ** 0.5)

            # ── re-shard and apply update ──
            if is_dtensor:
                ortho_sharded = distribute_tensor(ortho, grad.device_mesh, grad.placements)
                p.add_(ortho_sharded.to(p.dtype), alpha=-lr)
            else:
                p.add_(ortho.to(p.dtype), alpha=-lr)

    def _adamw_step(self, group):
        lr = group["lr"]
        b1, b2 = group["betas"]
        eps = group["eps"]
        wd  = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.float()

            state = self.state[p]
            if "step" not in state:
                state["step"] = torch.tensor(0.0)
                state["exp_avg"]    = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)

            state["step"] += 1
            t = state["step"].item()
            m = state["exp_avg"]
            v = state["exp_avg_sq"]

            m.mul_(b1).add_(grad, alpha=1 - b1)
            v.mul_(b2).addcmul_(grad, grad, value=1 - b2)

            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            step = (m / bc1) / ((v / bc2).sqrt() + eps)

            # weight decay + update
            p.mul_(1 - lr * wd)
            p.add_(step.to(p.dtype), alpha=-lr)
