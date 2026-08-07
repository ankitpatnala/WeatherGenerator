#!/usr/bin/env python
"""
Consecutive-step latent metrics: how each latent differs from its own previous step,
computed independently for two trajectories and overlaid.

  forecast[t] = FE(z_{t-1})            -- free-running rollout (one IC, autoregressive)
  encoder[t]  = FE(encode(real(t-1)))  -- 1-step forecast from consecutive real ICs

Both are post-FE latents valid at time t. For each stream, per step t (vs z_{t-1}):

  norm        ||z_t||_F
  cos         cos(z_t, z_{t-1})                 (global, flattened)  -> how much direction turns
  relchange   ||z_t - z_{t-1}||_F / ||z_t||_F                        -> magnitude of step change
  elem_frob   || (z_t - z_{t-1}) / z_{t-1} ||_F  (element-wise ratio) -> element-wise relative change

The free-running trajectory typically shows HIGHER cos-to-prev (under-evolves) and inflated
norm vs the truth-anchored one -- the latent-space signature of variance collapse / drift.

Usage:
  python compare_latents.py --forecast results/latent_drift_2015/setA \
                            --encoder  results/latent_drift_2015/setB \
                            [--out results/latent_drift_2015] [--eps 1e-3]
"""

import argparse
import glob
import json
import os

import numpy as np


def _iter_latents(d):
    """
    Yield (global_step, z) in order from chunked latent files.

    Supports the chunked layout z_{start:06d}.npy of shape (n, cells, dim), and falls back to
    the legacy one-file-per-step layout z_{idx:05d}.npy of shape (cells, dim).
    """
    files = sorted(glob.glob(os.path.join(d, "z_*.npy")))
    for f in files:
        arr = np.load(f, mmap_mode="r")
        start = int(os.path.basename(f).split("_")[1].split(".")[0])
        if arr.ndim == 3:  # chunked (n, cells, dim)
            for j in range(arr.shape[0]):
                yield start + j, np.asarray(arr[j], dtype=np.float32)
        else:  # legacy per-step (cells, dim)
            yield start, np.asarray(arr, dtype=np.float32)


def _stream_metrics(d, eps):
    """Consecutive-step metrics for one latent directory (chunked or per-step)."""
    norm, cos, relchange, elem_frob = [], [], [], []
    prev = None
    n = 0
    for t, z in _iter_latents(d):
        norm.append(np.linalg.norm(z))
        if prev is not None:
            diff = z - prev
            cos.append(float((z * prev).sum() / (np.linalg.norm(z) * np.linalg.norm(prev) + 1e-12)))
            relchange.append(np.linalg.norm(diff) / (np.linalg.norm(z) + 1e-12))
            denom = np.where(np.abs(prev) < eps, np.sign(prev) * eps + (prev == 0) * eps, prev)
            elem_frob.append(np.linalg.norm(diff / denom))
        else:
            cos.append(np.nan); relchange.append(np.nan); elem_frob.append(np.nan)
        prev = z
        if n % 100 == 0:
            print(f"  [{os.path.basename(d.rstrip('/'))}] t={n:4d}  norm={norm[-1]:.1f}  cos={cos[-1]:.4f}")
        n += 1
    return {
        "norm": np.array(norm), "cos": np.array(cos),
        "relchange": np.array(relchange), "elem_frob": np.array(elem_frob), "n": n,
    }


_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple", "tab:brown"]


def main():
    ap = argparse.ArgumentParser(
        description="Overlay consecutive-step latent metrics for one or more latent dirs."
    )
    # New N-series interface: repeat --series DIR LABEL for each trajectory.
    ap.add_argument("--series", nargs=2, action="append", metavar=("DIR", "LABEL"),
                    default=[], help="a latent dir and its legend label; repeatable")
    # Back-compat 2-series interface.
    ap.add_argument("--forecast", help="[legacy] first latent dir")
    ap.add_argument("--encoder", help="[legacy] second latent dir")
    ap.add_argument("--label-a", default="forecast")
    ap.add_argument("--label-b", default="encoder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="floor on |z_{t-1}| for the element-wise ratio metric")
    ap.add_argument("--trim", type=int, default=10,
                    help="drop the last N steps of each series (the encoder streams have a "
                         "dataloader final-batch artifact in the last ~8 samples)")
    args = ap.parse_args()

    series = list(args.series)
    if args.forecast:
        series.append([args.forecast, args.label_a])
    if args.encoder:
        series.append([args.encoder, args.label_b])
    if not series:
        ap.error("provide at least one --series DIR LABEL (or --forecast/--encoder)")

    out = args.out or os.path.dirname(series[0][0].rstrip("/"))
    os.makedirs(out, exist_ok=True)

    metrics = []
    for d, label in series:
        print(f"{label} ({d}):")
        m = _stream_metrics(d, args.eps)
        if args.trim > 0:  # drop the final-batch artifact tail
            keep = max(0, m["n"] - args.trim)
            for k in ("norm", "cos", "relchange", "elem_frob"):
                m[k] = m[k][:keep]
            m["n"] = keep
        metrics.append((label, m))

    panels = [
        ("norm", "||z_t||", "Latent Magnitude per Step"),
        ("cos", "Cosine Similarity", "Directional Drift  (1 = aligned, 0 = orthogonal)"),
        ("relchange", "||dz|| / ||z_t||", "Relative Change"),
        ("elem_frob", "|| (z_t-z_{t-1})/z_{t-1} ||_F",
         "Frobenius Norm of Element-wise Relative Change"),
    ]
    npz = {}
    for i, (label, m) in enumerate(metrics):
        for key in ("norm", "cos", "relchange", "elem_frob"):
            npz[f"s{i}_{key}"] = m[key]
    np.savez_compressed(os.path.join(out, "latent_step_metrics.npz"),
                        labels=np.array([lbl for lbl, _ in metrics]), **npz)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(len(panels), 1, figsize=(14, 16))
        for a, (key, ylab, title) in zip(ax, panels, strict=True):
            for i, (label, m) in enumerate(metrics):
                a.plot(m[key], color=_COLORS[i % len(_COLORS)], lw=1.0, label=label)
            if key == "cos":
                a.axhline(1.0, color="k", ls="--", lw=0.6)
            a.set_title(title); a.set_ylabel(ylab); a.set_xlabel("Step")
            a.legend(); a.grid(alpha=0.25)
        fig.tight_layout()
        png = os.path.join(out, "latent_step_metrics.png")
        fig.savefig(png, dpi=110)
        print(f"saved plot -> {png}")
    except Exception as e:
        print(f"[plot skipped] {e}")

    def _fin(x):
        v = x[np.isfinite(x)]
        return float(v[-1]) if v.size else None
    summary = {label: {k: _fin(m[k]) for k in ("norm", "cos", "relchange", "elem_frob")}
               | {"n": m["n"]} for label, m in metrics}
    with open(os.path.join(out, "latent_step_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
