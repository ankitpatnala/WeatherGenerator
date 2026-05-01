# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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
    Random Fourier Feature hashing + pointwise MLP for coordinate embedding.

    Each coordinate vector is projected onto `num_freqs` random frequencies,
    yielding a (sin, cos) pair per frequency — a unique high-frequency
    fingerprint distinguishing nearby spatial points unlike a linear or MLP
    projection. A shared pointwise MLP is applied to each (sin, cos) token
    independently (O(N × num_freqs), not O(N × num_freqs²) like attention),
    then mean-pooled to `dim_embed`. Nearby-point discrimination comes from
    the RFF step, so cross-frequency attention is unnecessary.

    The random projection matrix B is a frozen buffer; only the MLP weights
    are trained. Works for any dim_coord_in.
    """

    def __init__(
        self,
        dim_coord_in: int,
        dim_embed: int,
        num_freqs: int = 64,
        d_model: int = 64,
        num_layers: int = 2,
        sigma: float = 1.0,
    ):
        super().__init__()
        B = torch.randn(dim_coord_in, num_freqs) * sigma
        self.register_buffer("B", B)

        # Shared pointwise MLP applied independently to each frequency token.
        # Avoids O(N × num_freqs²) attention when N (target coord points) is
        # large; cross-token interaction is unnecessary since the RFF barcodes
        # already encode all discriminative spatial information per point.
        self.freq_mlp = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            *[layer for _ in range(num_layers - 1)
              for layer in (nn.Linear(d_model, d_model), nn.GELU())],
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, dim_embed)

    def reset_parameters(self):
        for m in self.freq_mlp.modules():
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
        tokens = self.freq_mlp(tokens)                          # (N, num_freqs, d_model)
        out = self.out_norm(tokens.mean(dim=1))                 # (N, d_model)  mean-pool freqs
        out = self.out_proj(out)                                # (N, dim_embed)
        return out.reshape(*prefix, -1)


class MultiResHashGridEmbedding(nn.Module):
    """
    Multi-resolution learnable spatial hash grid for (lat, lon) → dim_embed.
    Inspired by Instant-NGP (Müller et al., 2022).

    At each of `num_levels` log-spaced resolutions (coarsest_res → finest_res
    degrees), a hash table stores a small learnable feature vector per grid
    cell. A query coordinate is bilinearly interpolated between the 4
    surrounding vertices at each level; features from all levels are
    concatenated and projected to dim_embed via a small MLP.

    At the finest level every distinct ERA5 grid cell maps to near-unique hash
    entries — guaranteed per-cell discrimination without needing the hash size
    to equal the number of grid points.

    Expects x[..., 0] = lat (degrees, [-90, 90])
            x[..., 1] = lon (degrees, [-180, 180])
    Any extra coordinate dimensions are linearly projected and concatenated
    to the MLP input.
    """

    _P2 = 2654435761  # large prime for spatial hashing (from Instant-NGP)

    def __init__(
        self,
        dim_coord_in: int,
        dim_embed: int,
        num_levels: int = 8,
        feat_dim: int = 4,
        hash_size: int = 2**17,
        coarsest_res: float = 45.0,
        finest_res: float = 1.0,
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.feat_dim = feat_dim
        self.hash_size = hash_size
        self.dim_coord_extra = max(dim_coord_in - 2, 0)

        # Log-spaced grid resolutions from coarsest to finest (degrees)
        log_ratio = math.log(finest_res / coarsest_res) / max(num_levels - 1, 1)
        resolutions = [coarsest_res * math.exp(log_ratio * l) for l in range(num_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.float32))

        # One learned hash table per level: hash_size entries × feat_dim
        self.hash_tables = nn.ModuleList(
            [nn.Embedding(hash_size, feat_dim) for _ in range(num_levels)]
        )

        # Optional projection for extra coordinate dimensions
        dim_mlp_in = num_levels * feat_dim
        if self.dim_coord_extra > 0:
            self.extra_proj = nn.Linear(self.dim_coord_extra, mlp_hidden)
            dim_mlp_in += mlp_hidden
        else:
            self.extra_proj = None

        # MLP: concatenated hash features → dim_embed
        layers: list[nn.Module] = [nn.Linear(dim_mlp_in, mlp_hidden), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU()]
        layers.append(nn.Linear(mlp_hidden, dim_embed))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for emb in self.hash_tables:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

    def reset_parameters(self):
        self._init_weights()
        if self.extra_proj is not None:
            self.extra_proj.reset_parameters()
        for m in self.mlp:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def _hash(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        return ((i ^ (j * self._P2)) % self.hash_size).long()

    def _interpolate(self, lat: torch.Tensor, lon: torch.Tensor, level: int) -> torch.Tensor:
        res = self.resolutions[level].item()

        # Map lat [-90,90] → [0, 180/res], lon [-180,180] → [0, 360/res]
        lat_g = (lat + 90.0) / res
        lon_g = (lon + 180.0) / res

        i0 = lat_g.floor().long()
        j0 = lon_g.floor().long()
        fi = (lat_g - i0.float()).unsqueeze(-1)  # (N, 1)
        fj = (lon_g - j0.float()).unsqueeze(-1)

        i1 = i0 + 1
        j1 = j0 + 1

        lat_cells = int(math.ceil(180.0 / res))
        lon_cells = int(math.ceil(360.0 / res))

        i0 = i0.clamp(0, lat_cells)
        i1 = i1.clamp(0, lat_cells)
        j0 = j0 % lon_cells          # longitude wraps
        j1 = j1 % lon_cells

        table = self.hash_tables[level]
        f00 = table(self._hash(i0, j0))  # (N, feat_dim)
        f01 = table(self._hash(i0, j1))
        f10 = table(self._hash(i1, j0))
        f11 = table(self._hash(i1, j1))

        # Bilinear interpolation: first along lon, then along lat
        f_bot = f00 * (1.0 - fj) + f01 * fj
        f_top = f10 * (1.0 - fj) + f11 * fj
        return f_bot * (1.0 - fi) + f_top * fi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim_coord_in), first two dims are (lat, lon) in degrees
        prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])           # (N, dim_coord_in)

        lat = x_flat[:, 0]
        lon = x_flat[:, 1]

        feats = [self._interpolate(lat, lon, l) for l in range(self.num_levels)]
        mlp_in = torch.cat(feats, dim=-1)              # (N, num_levels * feat_dim)

        if self.extra_proj is not None:
            extra = self.extra_proj(x_flat[:, 2:])    # (N, mlp_hidden)
            mlp_in = torch.cat([mlp_in, extra], dim=-1)

        out = self.mlp(mlp_in)                         # (N, dim_embed)
        return out.reshape(*prefix, -1)
