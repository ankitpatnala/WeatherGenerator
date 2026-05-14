"""
Temporal variance of latents: var over N time steps per (12288, 2048) element.

Usage:
  python variance_analysis.py
  python variance_analysis.py --folder_a /path/enc/*.pt --folder_b /path/fc/*.pt
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch


BASE = "/p/scratch/weatherai/slurm/slurm_weathergen_atmosfo2_copy_dir/WeatherGenerator"


def load_latents(folder_pattern: str, num_latents: int, rng: np.random.Generator) -> torch.Tensor:
    files = sorted(
        glob.glob(folder_pattern),
        key=lambda f: int(re.search(r"\d+", os.path.basename(f)).group()),
    )
    if len(files) < num_latents:
        raise ValueError(f"Only {len(files)} files found, need {num_latents}.")
    chosen = rng.choice(len(files), num_latents, replace=False)
    latents = []
    for i in chosen:
        data = torch.load(files[i], map_location="cpu")
        t = data["latent"] if isinstance(data, dict) else data
        latents.append(t.squeeze(0).float())   # [12288, 2048]
    return torch.stack(latents)                # [N, 12288, 2048]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_a", default=f"{BASE}/latents_2/*.pt",
                        help="Encoder latents (dict with 'latent' key)")
    parser.add_argument("--folder_b", default=f"{BASE}/latents/*.pt",
                        help="Forecast latents (raw tensor)")
    parser.add_argument("--label_a",    default="encoder")
    parser.add_argument("--label_b",    default="forecast")
    parser.add_argument("--num_latents", type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--output_dir",  default="./plots/variance")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.num_latents} latents [{args.label_a}] ...")
    lat_a = load_latents(args.folder_a, args.num_latents, rng)  # [N, 12288, 2048]
    var_a = lat_a.var(dim=0).numpy()                             # [12288, 2048]
    print(f"  variance shape: {var_a.shape}  min={var_a.min():.4f}  max={var_a.max():.4f}")

    print(f"Loading {args.num_latents} latents [{args.label_b}] ...")
    lat_b = load_latents(args.folder_b, args.num_latents, rng)
    var_b = lat_b.var(dim=0).numpy()
    print(f"  variance shape: {var_b.shape}  min={var_b.min():.4f}  max={var_b.max():.4f}")

    # shared colour scale (log1p)
    combined  = np.concatenate([var_a.ravel(), var_b.ravel()])
    vmin = np.log1p(np.percentile(combined, 1))
    vmax = np.log1p(np.percentile(combined, 99))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    def _show(ax, data, title, vmin, vmax, cmap="viridis"):
        im = ax.imshow(np.log1p(data), aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("embedding dim  (0 → 2047)")
        ax.set_ylabel("spatial position  (0 → 12287)")
        return im

    # left: forecast, middle: encoder, right: log ratio
    im_b = _show(axes[0], var_b, f"{args.label_b}  [log1p scale]", vmin, vmax)
    im_a = _show(axes[1], var_a, f"{args.label_a}  [log1p scale]", vmin, vmax)

    ratio = np.log(var_b + 1e-12) - np.log(var_a + 1e-12)   # positive = forecast higher
    r_lim = np.percentile(np.abs(ratio), 98)
    im_r  = axes[2].imshow(ratio, aspect="auto", interpolation="nearest",
                           cmap="RdBu_r", vmin=-r_lim, vmax=r_lim)
    axes[2].set_title(f"log ratio  ({args.label_b} / {args.label_a})\nred = forecast higher  |  blue = encoder higher")
    axes[2].set_xlabel("embedding dim  (0 → 2047)")
    axes[2].set_ylabel("spatial position  (0 → 12287)")

    plt.colorbar(im_b, ax=axes[0], fraction=0.02, pad=0.02, label="log1p(var)")
    plt.colorbar(im_a, ax=axes[1], fraction=0.02, pad=0.02, label="log1p(var)")
    plt.colorbar(im_r, ax=axes[2], fraction=0.02, pad=0.02, label="log ratio")

    plt.suptitle("Temporal Variance per element  [12288 × 2048]", fontsize=13)
    plt.tight_layout()

    out = os.path.join(args.output_dir, "temporal_variance.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()

    np.save(os.path.join(args.output_dir, f"var_{args.label_a}.npy"), var_a)
    np.save(os.path.join(args.output_dir, f"var_{args.label_b}.npy"), var_b)
    print(f"Saved: {args.output_dir}/var_*.npy")


if __name__ == "__main__":
    main()
