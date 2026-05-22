# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import torch
import torch.nn as nn

from weathergen.model.norms import AdaLayerNorm, RMSNorm


class NamedLinear(torch.nn.Module):
    def __init__(self, name: str | None = None, **kwargs):
        super(NamedLinear, self).__init__()
        self.linear = nn.Linear(**kwargs)
        if name is not None:
            self.name = name

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_layers=2,
        hidden_factor=2,
        pre_layer_norm=True,
        dropout_rate=0.0,
        nonlin=torch.nn.GELU,
        with_residual=False,
        norm_type="LayerNorm",
        dim_aux=None,
        norm_eps=1e-5,
        name: str | None = None,
    ):
        """Constructor"""

        super(MLP, self).__init__()

        if name is not None:
            self.name = name

        assert num_layers >= 2

        self.with_residual = with_residual
        self.with_aux = dim_aux is not None
        dim_hidden = int(dim_in * hidden_factor)

        self.layers = torch.nn.ModuleList()

        norm = torch.nn.LayerNorm if norm_type == "LayerNorm" else RMSNorm

        if pre_layer_norm:
            self.layers.append(
                norm(dim_in, eps=norm_eps)
                if dim_aux is None
                else AdaLayerNorm(dim_in, dim_aux, norm_eps=norm_eps)
            )

        self.layers.append(torch.nn.Linear(dim_in, dim_hidden))
        self.layers.append(nonlin())
        self.layers.append(torch.nn.Dropout(p=dropout_rate))

        for _ in range(num_layers - 2):
            self.layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nonlin())
            self.layers.append(torch.nn.Dropout(p=dropout_rate))

        self.layers.append(torch.nn.Linear(dim_hidden, dim_out))

    def forward(self, *args):
        x, x_in, aux = args[0], args[0], args[-1]

        for i, layer in enumerate(self.layers):
            x = layer(x, aux) if (i == 0 and self.with_aux) else layer(x)

        if self.with_residual:
            if x.shape[-1] == x_in.shape[-1]:
                x = x_in + x
            else:
                assert x.shape[-1] % x_in.shape[-1] == 0
                x = x + x_in.repeat([*[1 for _ in x.shape[:-1]], x.shape[-1] // x_in.shape[-1]])

        return x


class SwiGLUCoordEmbed(torch.nn.Module):
    """
    3-layer SwiGLU coordinate embedding.

    Each layer: SiLU(x @ W_gate) ⊙ (x @ W_val) → x @ W_proj
    Layers 2-3 add a pre-norm residual so gradients flow cleanly.
    Compared to a GELU-MLP, the multiplicative gate creates sharp
    input-dependent boundaries — nearby coordinates diverge quickly
    in embedding space even when their raw features are nearly identical.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_factor: int = 4,
        norm_eps: float = 1e-5,
        name: str | None = None,
    ):
        super().__init__()
        if name is not None:
            self.name = name

        H = dim_out * hidden_factor
        act = nn.SiLU

        # layer 1: dim_in → dim_out  (no residual — dims differ)
        self.norm1   = nn.LayerNorm(dim_in, eps=norm_eps)
        self.gate1   = nn.Linear(dim_in, H, bias=False)
        self.val1    = nn.Linear(dim_in, H, bias=False)
        self.proj1   = nn.Linear(H, dim_out, bias=False)

        # layer 2: dim_out → dim_out  (pre-norm + residual)
        self.norm2   = nn.LayerNorm(dim_out, eps=norm_eps)
        self.gate2   = nn.Linear(dim_out, H, bias=False)
        self.val2    = nn.Linear(dim_out, H, bias=False)
        self.proj2   = nn.Linear(H, dim_out, bias=False)

        # layer 3: dim_out → dim_out  (pre-norm + residual)
        self.norm3   = nn.LayerNorm(dim_out, eps=norm_eps)
        self.gate3   = nn.Linear(dim_out, H, bias=False)
        self.val3    = nn.Linear(dim_out, H, bias=False)
        self.proj3   = nn.Linear(H, dim_out, bias=False)

        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # layer 1
        h = self.norm1(x)
        x = self.proj1(self.act(self.gate1(h)) * self.val1(h))

        # layer 2 (residual)
        h = self.norm2(x)
        x = x + self.proj2(self.act(self.gate2(h)) * self.val2(h))

        # layer 3 (residual)
        h = self.norm3(x)
        x = x + self.proj3(self.act(self.gate3(h)) * self.val3(h))

        return x
