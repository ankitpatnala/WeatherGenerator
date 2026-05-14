"""
Manifold analysis of encoder / forecast-engine latents.

Metrics per sampled spatial position (shape: [N, 2048]):
  - TwoNN intrinsic dimension  (local, assumption-free)
  - PCA effective rank         (global, participation ratio)
  - Log-volume ratio           (Σ log σ_i  vs  D * log σ_mean)
  - Fraction of variance in top-k PCs

Usage:
  python manifold_analysis.py \
      --folder_a /path/to/encoder_latents/*.pt \
      --folder_b /path/to/forecast_latents/*.pt \
      --num_latents 500 \
      --num_spatial 256 \
      --output_dir ./plots/manifold
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_single(path: str) -> torch.Tensor:
    """
    Handles two on-disk formats:
      - dict  {'idx': scalar, 'latent': [1, 12288, 2048]}  (latents_2 / encoder)
      - raw tensor  [1, 12288, 2048]                        (latents / forecast)
    Returns [12288, 2048].
    """
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        return data["latent"].squeeze(0).float()
    return data.squeeze(0).float()


def load_latents(folder_pattern: str, num_latents: int, rng: np.random.Generator):
    files = sorted(
        glob.glob(folder_pattern),
        key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group()),
    )
    if len(files) < num_latents:
        raise ValueError(f"Only {len(files)} files found, need {num_latents}.")

    chosen = rng.choice(len(files), num_latents, replace=False)
    latents = [_load_single(files[i]) for i in chosen]
    return torch.stack(latents)  # [N, 12288, 2048]


# ---------------------------------------------------------------------------
# Per-position metrics — all computed from a single SVD
# ---------------------------------------------------------------------------

def _svd_once(X: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre X, run randomised SVD once. Returns (X_c, S, Vt) as numpy arrays."""
    from sklearn.utils.extmath import randomized_svd
    X_c = (X - X.mean(0)).numpy().astype(np.float32)
    n_components = min(X_c.shape) - 1
    _, S, Vt = randomized_svd(X_c, n_components=n_components, random_state=0)
    return X_c, S, Vt


def _log_ball_volume(d: int) -> float:
    return (d / 2) * np.log(np.pi) - gammaln(d / 2 + 1)


def _metrics_from_svd(X_c, S, Vt, k_var: int, hull_k: int, with_hull: bool):
    N = X_c.shape[0]
    S = S[S > 1e-10]

    # effective rank
    p = S / S.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))

    # log-volume ratio
    log_vol_data = np.log(S).sum()
    log_vol_ref  = len(S) * np.log(S.mean())
    log_vol = float(log_vol_data / log_vol_ref) if log_vol_ref != 0 else 0.0

    # variance explained by top-k_var PCs
    var_exp = float((S[:k_var] ** 2).sum() / (S ** 2).sum())

    # TwoNN on PCA-projected data (top min(N//2, 100) dims — much faster than 2048 dims)
    pca_k = min(N // 2, 100, len(S))
    X_proj = X_c @ Vt[:pca_k].T                       # [N, pca_k]
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X_proj)
    dists, _ = nbrs.kneighbors(X_proj)
    r1, r2 = dists[:, 1], dists[:, 2]
    mask = (r1 > 0) & (r2 > 0)
    twonn_dim = float(1.0 / np.log(r2[mask] / r1[mask]).mean())

    # convex hull in top-hull_k PCA subspace (optional, slow for hull_k > 8)
    hull_vol, hull_ratio = float("nan"), float("nan")
    if with_hull and hull_k <= len(S):
        X_hull = X_c @ Vt[:hull_k].T
        log_semi = np.log(S[:hull_k]) - 0.5 * np.log(N)
        ellipsoid_vol = np.exp(_log_ball_volume(hull_k) + log_semi.sum())
        try:
            hv = ConvexHull(X_hull).volume
            hull_vol  = hv
            hull_ratio = hv / ellipsoid_vol if ellipsoid_vol > 0 else float("nan")
        except QhullError:
            pass

    return eff_rank, log_vol, var_exp, twonn_dim, hull_vol, hull_ratio


# ---------------------------------------------------------------------------
# Main analysis over sampled spatial positions
# ---------------------------------------------------------------------------

def analyse(
    latents: torch.Tensor,  # [N, 12288, 2048]
    num_spatial: int,
    k_var: int,
    hull_k: int,
    with_hull: bool,
    rng: np.random.Generator,
    label: str,
) -> dict:
    N, S, D = latents.shape
    spatial_idx = rng.choice(S, num_spatial, replace=False)

    eff_ranks, log_vols, var_exp, twonn_dims, hull_vols, hull_ratios = [], [], [], [], [], []

    for i, si in enumerate(spatial_idx):
        X = latents[:, si, :]                          # [N, 2048]
        X_c, Sv, Vt = _svd_once(X)                    # single SVD reused by all metrics
        er, lv, ve, td, hv, hr = _metrics_from_svd(X_c, Sv, Vt, k_var, hull_k, with_hull)

        eff_ranks.append(er)
        log_vols.append(lv)
        var_exp.append(ve)
        twonn_dims.append(td)
        hull_vols.append(hv)
        hull_ratios.append(hr)

        if (i + 1) % 16 == 0:
            print(f"  [{label}] {i+1}/{num_spatial} positions done")

    return {
        "eff_rank":    np.array(eff_ranks),
        "log_vol":     np.array(log_vols),
        "var_exp":     np.array(var_exp),
        "twonn_dim":   np.array(twonn_dims),
        "hull_volume": np.array(hull_vols),
        "hull_ratio":  np.array(hull_ratios),
    }


def print_summary(stats: dict, label: str):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for key, vals in stats.items():
        print(f"  {key:12s}:  mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"median={np.median(vals):.3f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_TITLES = {
    "eff_rank":    "Effective Rank (participation ratio)",
    "log_vol":     "Log-Volume Ratio (data / isotropic ref)",
    "var_exp":     "Variance Explained (top-k PCs)",
    "twonn_dim":   "TwoNN Intrinsic Dimension",
    "hull_volume": "Convex Hull Volume (top-k PCA subspace)",
    "hull_ratio":  "Hull / Ellipsoid Volume Ratio (top-k PCA subspace)",
}


def _make_axes(n: int):
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    return fig, axes.flatten()


def plot_comparison(stats_a: dict, stats_b: dict, label_a: str, label_b: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metrics = list(stats_a.keys())
    fig, axes = _make_axes(len(metrics))

    for ax, key in zip(axes, metrics):
        a = stats_a[key]
        b = stats_b[key]
        # drop nans for hull metrics
        a_clean = a[np.isfinite(a)]
        b_clean = b[np.isfinite(b)]
        ax.hist(a_clean, bins=30, alpha=0.6, label=label_a, color="steelblue")
        ax.hist(b_clean, bins=30, alpha=0.6, label=label_b, color="tomato")
        if len(a_clean):
            ax.axvline(a_clean.mean(), color="steelblue", linestyle="--", linewidth=1.5,
                       label=f"{label_a} μ={a_clean.mean():.3f}")
        if len(b_clean):
            ax.axvline(b_clean.mean(), color="tomato", linestyle="--", linewidth=1.5,
                       label=f"{label_b} μ={b_clean.mean():.3f}")
        ax.set_title(METRIC_TITLES.get(key, key))
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    plt.suptitle("Manifold Analysis: Encoder vs Forecast Latents", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "manifold_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    plt.close()


def plot_single(stats: dict, label: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metrics = list(stats.keys())
    fig, axes = _make_axes(len(metrics))

    for ax, key in zip(axes, metrics):
        vals = stats[key]
        vals_clean = vals[np.isfinite(vals)]
        ax.hist(vals_clean, bins=30, color="steelblue", alpha=0.8)
        if len(vals_clean):
            ax.axvline(vals_clean.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"mean={vals_clean.mean():.3f}")
        ax.set_title(METRIC_TITLES.get(key, key))
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend()

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    plt.suptitle(f"Manifold Analysis: {label}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "manifold_analysis.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Temporal variance analysis  — var over N time steps per (spatial, embed) element
# ---------------------------------------------------------------------------

def compute_temporal_variance(latents: torch.Tensor) -> torch.Tensor:
    """latents: [N, 12288, 2048]  →  variance [12288, 2048]"""
    return latents.var(dim=0)   # [S, D]


def _imshow_var(ax, var_np: np.ndarray, title: str, vmin=None, vmax=None, log=True):
    """Show [12288, 2048] variance as a 2-D heatmap."""
    data = np.log1p(var_np) if log else var_np
    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=vmin if vmin is not None else np.percentile(data, 1),
        vmax=vmax if vmax is not None else np.percentile(data, 99),
    )
    ax.set_title(title)
    ax.set_xlabel("embedding dim  (0 → 2047)")
    ax.set_ylabel("spatial position  (0 → 12287)")
    return im


def plot_temporal_variance(var_a: torch.Tensor, label_a: str,
                           var_b: torch.Tensor | None, label_b: str,
                           output_dir: str):
    """
    Show per-element temporal variance as [12288, 2048] heatmaps (log scale).
    Single label_a: 1 panel.
    Two labels: 3 panels — a, b, ratio b/a.
    """
    os.makedirs(output_dir, exist_ok=True)
    var_a_np = var_a.float().numpy()          # [12288, 2048]

    if var_b is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        im = _imshow_var(ax, var_a_np, f"Temporal variance — {label_a}  [log1p scale]")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="log1p(variance)")
        plt.tight_layout()
    else:
        var_b_np = var_b.float().numpy()      # [12288, 2048]

        # shared colour scale for a and b
        combined = np.concatenate([var_a_np.ravel(), var_b_np.ravel()])
        vmin = np.log1p(np.percentile(combined, 1))
        vmax = np.log1p(np.percentile(combined, 99))

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        im_a = _imshow_var(axes[0], var_a_np,
                           f"{label_a}  [log1p]", vmin=vmin, vmax=vmax)
        im_b = _imshow_var(axes[1], var_b_np,
                           f"{label_b}  [log1p]", vmin=vmin, vmax=vmax)

        # ratio: log(var_b / var_a) — centred at 0
        ratio = np.log(var_b_np + 1e-12) - np.log(var_a_np + 1e-12)
        r_lim = np.percentile(np.abs(ratio), 98)
        im_r = axes[2].imshow(ratio, aspect="auto", interpolation="nearest",
                              cmap="RdBu_r", vmin=-r_lim, vmax=r_lim)
        axes[2].set_title(f"log ratio  ({label_b} / {label_a})\nblue=forecast lower, red=forecast higher")
        axes[2].set_xlabel("embedding dim  (0 → 2047)")
        axes[2].set_ylabel("spatial position  (0 → 12287)")

        plt.colorbar(im_a, ax=axes[0], fraction=0.02, pad=0.02, label="log1p(var)")
        plt.colorbar(im_b, ax=axes[1], fraction=0.02, pad=0.02, label="log1p(var)")
        plt.colorbar(im_r, ax=axes[2], fraction=0.02, pad=0.02, label="log ratio")

        plt.suptitle("Temporal Variance per element  [12288 × 2048]", fontsize=13)
        plt.tight_layout()

    path = os.path.join(output_dir, "temporal_variance.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # save raw arrays
    np.save(os.path.join(output_dir, f"var_{label_a}.npy"), var_a_np)
    if var_b is not None:
        np.save(os.path.join(output_dir, f"var_{label_b}.npy"), var_b_np)
    print(f"Saved variance arrays to {output_dir}/var_*.npy")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    base = "/p/scratch/weatherai/slurm/slurm_weathergen_atmosfo2_copy_dir/WeatherGenerator"
    parser.add_argument("--folder_a", type=str,
                        default=f"{base}/latents_2/*.pt",
                        help="Glob for encoder latent .pt files (dict format with 'latent' key)")
    parser.add_argument("--folder_b", type=str,
                        default=f"{base}/latents/*.pt",
                        help="Glob for forecast latent .pt files (raw tensor format)")
    parser.add_argument("--label_a", type=str, default="encoder")
    parser.add_argument("--label_b", type=str, default="forecast")
    parser.add_argument("--num_latents", type=int, default=500,
                        help="Number of latent files to sample")
    parser.add_argument("--num_spatial", type=int, default=256,
                        help="Number of spatial positions to sample per analysis")
    parser.add_argument("--k_var", type=int, default=81,
                        help="Number of PCs for variance-explained metric")
    parser.add_argument("--hull_k", type=int, default=5,
                        help="PCA dims for convex hull (keep <=8 for speed)")
    parser.add_argument("--with_hull", action="store_true",
                        help="Compute convex hull (slow, off by default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./plots/manifold")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.num_latents} latents from: {args.folder_a}")
    latents_a = load_latents(args.folder_a, args.num_latents, rng)
    print(f"  shape: {tuple(latents_a.shape)}")

    print(f"\nAnalysing {args.num_spatial} spatial positions [{args.label_a}] ...")
    print(f"  hull={'on, k=' + str(args.hull_k) if args.with_hull else 'off (--with_hull to enable)'}")
    stats_a = analyse(latents_a, args.num_spatial, args.k_var, args.hull_k, args.with_hull, rng, args.label_a)
    print_summary(stats_a, args.label_a)

    if args.folder_b:
        print(f"\nLoading {args.num_latents} latents from: {args.folder_b}")
        latents_b = load_latents(args.folder_b, args.num_latents, rng)
        print(f"  shape: {tuple(latents_b.shape)}")

        print(f"\nAnalysing {args.num_spatial} spatial positions [{args.label_b}] ...")
        stats_b = analyse(latents_b, args.num_spatial, args.k_var, args.hull_k, args.with_hull, rng, args.label_b)
        print_summary(stats_b, args.label_b)

        plot_comparison(stats_a, stats_b, args.label_a, args.label_b, args.output_dir)

        print("\nComputing temporal variance ...")
        var_a = compute_temporal_variance(latents_a)
        var_b = compute_temporal_variance(latents_b)
        plot_temporal_variance(var_a, args.label_a, var_b, args.label_b, args.output_dir)
    else:
        plot_single(stats_a, args.label_a, args.output_dir)

        print("\nComputing temporal variance ...")
        var_a = compute_temporal_variance(latents_a)
        plot_temporal_variance(var_a, args.label_a, None, "", args.output_dir)


if __name__ == "__main__":
    main()
