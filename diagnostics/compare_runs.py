#!/usr/bin/env python
"""
Compare free-running rollouts of the same model under different prescribed SST offsets.

Answers two questions the animations cannot:
  1. Does the warming response DRIFT? Per-step global-mean of (run - ref) over the whole
     rollout separates "equilibrated early and held" from "kept climbing / decayed", which is
     what distinguishes a genuine equilibrium response from the run wandering off the model's
     training distribution.
  2. What colour range makes the absolute fields comparable? Reports the min/max over ALL runs
     together, to be passed to animate_decadal.py --vmin/--vmax so the videos share a scale.

Usage:
  python diagnostics/compare_runs.py --ref sst_exp1_1y_0K \
      --run sst_exp1_1y_2K --run sst_exp1_1y_4K --field 2t

Writes <out-dir>/<field>_global_mean.csv and <field>_drift.png.

Note on the mean: the o96 octahedral reduced Gaussian grid is quasi-equal-area (rows thin out
toward the poles), so the unweighted point mean already approximates the area mean -- applying
cos(lat) weights on top of it would double-count the convergence and bias the result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_io import load_field, open_rollout, resolve_zip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="baseline run id (subtracted from each --run)")
    ap.add_argument("--run", action="append", required=True, help="run id; repeatable")
    ap.add_argument("--field", default="2t")
    ap.add_argument("--stream", default="ERA5")
    ap.add_argument("--source", default="prediction", choices=["prediction", "target"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out-dir", default="results/comparison")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ref] {args.ref}")
    ref_src = open_rollout([resolve_zip(args.ref)], args.stream)
    steps = sorted(ref_src)[:: args.stride]
    if args.max_frames:
        steps = steps[: args.max_frames]

    ref, lat, _lon, times = load_field(ref_src, args.field, steps, args.source)

    series, ranges = {}, [(float(np.nanmin(ref)), float(np.nanmax(ref)))]
    for run in args.run:
        print(f"[run] {run}")
        src = open_rollout([resolve_zip(run)], args.stream)
        missing = [s for s in steps if s not in src]
        if missing:
            raise ValueError(f"{run} is missing {len(missing)} steps (first: {missing[:5]})")
        vals, lat_r, _, times_r = load_field(src, args.field, steps, args.source)
        if vals.shape != ref.shape:
            raise ValueError(f"{run} shape {vals.shape} != ref {ref.shape}; refusing to compare")
        if not np.allclose(lat, lat_r, atol=1e-5):
            raise ValueError(f"{run} is on a different grid than the reference")
        bad = int((times != times_r).sum())
        if bad:
            raise ValueError(f"{run} valid times differ from the reference at {bad} steps")
        ranges.append((float(np.nanmin(vals)), float(np.nanmax(vals))))
        # per-step global mean of the difference field = the forced response through time
        series[run] = np.nanmean(vals - ref, axis=1)

    vmin = min(r[0] for r in ranges)
    vmax = max(r[1] for r in ranges)
    print(f"\nSHARED COLOUR RANGE over {[args.ref, *args.run]}: "
          f"--vmin {vmin:.4g} --vmax {vmax:.4g}")

    csv = out_dir / f"{args.field}_global_mean.csv"
    with csv.open("w") as fh:
        fh.write("step,time," + ",".join(f"{r}_minus_{args.ref}" for r in args.run) + "\n")
        for i, s in enumerate(steps):
            row = ",".join(f"{series[r][i]:.6f}" for r in args.run)
            fh.write(f"{s},{str(times[i])[:19]},{row}\n")
    print(f"wrote {csv}")

    # per-run summary: early (first 200 steps) vs late (last 200) says whether it drifted
    print(f"\n{'run':<24}{'all':>10}{'first200':>10}{'last200':>10}{'drift':>10}")
    for run in args.run:
        s_ = series[run]
        early, late = float(s_[:200].mean()), float(s_[-200:].mean())
        print(f"{run:<24}{s_.mean():>10.4f}{early:>10.4f}{late:>10.4f}{late - early:>+10.4f}")

    _plot(out_dir, args, steps, times, series)


def _plot(out_dir, args, steps, times, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5), dpi=140)
    days = np.arange(len(steps)) * 0.25  # 6-hourly steps
    for run, s_ in series.items():
        line, = ax.plot(days, s_, lw=0.7, alpha=0.45)
        # 30-day running mean: the per-step signal is dominated by synoptic noise
        w = min(120, max(1, len(s_) // 10))
        smooth = np.convolve(s_, np.ones(w) / w, mode="valid")
        ax.plot(days[w - 1:], smooth, lw=2.0, color=line.get_color(),
                label=f"{run} - {args.ref}  (mean {s_.mean():+.3f} K)")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("forecast lead time (days)")
    ax.set_ylabel(f"global-mean {args.field} difference (K)")
    ax.set_title(f"Forced response through the rollout: {args.field}, baseline {args.ref}\n"
                 f"(thin = per step, thick = {w // 4}-day running mean)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = out_dir / f"{args.field}_drift.png"
    fig.savefig(png)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
