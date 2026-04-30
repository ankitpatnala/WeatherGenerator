# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import math

import math

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


class FourierHashCoordEmbedding(nn.Module):
    """
    Random Fourier Feature hashing + mini-BERT for coordinate embedding.

    Each coordinate vector is projected onto `num_freqs` random frequencies,
    yielding a (sin, cos) pair per frequency — a unique high-frequency
    fingerprint that distinguishes nearby spatial points unlike a linear or
    MLP projection. A small pre-norm self-attention stack (BERT-style) then
    learns cross-frequency dependencies before mean-pooling to `dim_embed`.

    The random projection matrix B is a frozen buffer (not learned); only the
    transformer weights are trained.
    """

    def __init__(
        self,
        dim_coord_in: int,
        dim_embed: int,
        num_freqs: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        sigma: float = 1.0,
    ):
        super().__init__()
        # Fixed random projection: dim_coord_in → num_freqs scalar projections
        B = torch.randn(dim_coord_in, num_freqs) * sigma
        self.register_buffer("B", B)

        # Project each 2D (sin, cos) token up to transformer width
        self.token_proj = nn.Linear(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable than post-norm)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, dim_embed)

    def reset_parameters(self):
        self.token_proj.reset_parameters()
        for m in self.transformer.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.out_norm.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim_coord_in)
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])                    # (N, dim_coord_in)

        proj = 2.0 * math.pi * (x_flat @ self.B)               # (N, num_freqs)
        tokens = torch.stack([proj.sin(), proj.cos()], dim=-1)  # (N, num_freqs, 2)
        tokens = self.token_proj(tokens)                        # (N, num_freqs, d_model)
        tokens = self.transformer(tokens)                       # (N, num_freqs, d_model)
        out = self.out_norm(tokens.mean(dim=1))                 # (N, d_model)
        out = self.out_proj(out)                                # (N, dim_embed)
        return out.reshape(*prefix, -1)
