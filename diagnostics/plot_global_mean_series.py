#!/usr/bin/env python
"""
Global-mean time series of one field for several rollouts, plus the ERA5 target.

One line per run (and one for the truth), each point the global mean of that field at that
forecast step. This is the absolute view that complements the difference animations: it shows
the seasonal cycle, whether the runs track the truth, and whether any of them drifts away.

Unlike the map animations this needs NO canonical grid reordering: a mean over all points is
permutation-invariant, so each step is read and reduced immediately and nothing large is held.

Usage:
  python diagnostics/plot_global_mean_series.py --run sst_exp1_1y_0K --run sst_exp1_1y_2K \
      --run sst_exp1_1y_4K --field 2t

Note on the mean: the o96 octahedral reduced Gaussian grid is quasi-equal-area (rows thin out
toward the poles), so an unweighted point mean already approximates the area mean -- cos(lat)
weights on top of it would double-count the convergence.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_io import open_rollout, resolve_zip  # noqa: E402


REGIONS = ("global", "nh", "sh")


def _region_masks(coords, cache, equator_band=0.5):
    """
    Boolean masks for each region given ONE step's coords, cached by coords layout.

    The point order is not constant across steps, so a mask built for one step cannot be
    reused blindly for another -- it is keyed on the coords themselves.

    The o96 grid has no row exactly at the equator (nearest rows sit at +/-0.4675 deg), but
    those two rows straddle it and belong cleanly to neither hemisphere, so `equator_band`
    excludes |lat| < band from the hemispheric means. `global` deliberately keeps every point
    so it remains the true global mean.
    """
    key = np.asarray(coords, dtype=np.float32).tobytes()[:4096]
    m = cache.get(key)
    if m is None:
        lat = np.asarray(coords)[:, 0]
        off_equator = np.abs(lat) >= equator_band
        m = {"global": slice(None), "nh": (lat > 0) & off_equator, "sh": (lat < 0) & off_equator}
        cache[key] = m
    return m


def mean_series(src, field, steps, group, regions=REGIONS, equator_band=0.5):
    """Per-step area mean of `field` per region, or None if that group is absent."""
    root = src[steps[0]]
    if group not in root[str(steps[0])]:
        return None
    ch = list(root[f"{steps[0]}/{group}"].attrs["channels"]).index(field)
    has_ens = root[f"{steps[0]}/{group}/data"].ndim == 3

    out = {r: np.full(len(steps), np.nan, dtype=np.float64) for r in regions}
    times = np.empty(len(steps), dtype="datetime64[ns]")
    mask_cache = {}
    _reported = False
    for i, s in enumerate(steps):
        nd = src[s][f"{s}/{group}"]
        d = nd["data"]
        v = np.asarray(d[:, ch, 0] if has_ens else d[:, ch])
        masks = _region_masks(np.asarray(nd["coords"]), mask_cache, equator_band)
        if not _reported:
            n_tot = len(v)
            counts = {r: (n_tot if isinstance(masks[r], slice) else int(masks[r].sum()))
                      for r in regions}
            print(f"    points per region (of {n_tot}): {counts}"
                  f"  [excluded |lat| < {equator_band} deg from nh/sh]", flush=True)
            _reported = True
        for r in regions:
            out[r][i] = np.nanmean(v[masks[r]])
        times[i] = np.asarray(nd["times"][0])
        if i % 500 == 0:
            print(f"    [{group}] {i}/{len(steps)}", flush=True)
    return out, times


def _read_csv(path):
    """Read back a series CSV written by this script -> (times, {name: values})."""
    import csv as _csv

    with open(path) as fh:
        rows = list(_csv.reader(fh))
    header, body = rows[0], rows[1:]
    names = header[2:]
    times = np.array([np.datetime64(r[1]) for r in body])
    series = {n: np.array([float(r[2 + j]) for r in body]) for j, n in enumerate(names)}
    return times, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="run id; repeatable")
    ap.add_argument("--field", default="2t")
    ap.add_argument("--stream", default="ERA5")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-target", action="store_true", help="skip the ERA5 truth line")
    ap.add_argument("--out-dir", default="results/comparison")
    ap.add_argument("--smooth-days", type=float, default=0,
                    help="overlay a running mean of this many days on top of the raw 6-hourly "
                         "series (0 = off). The raw signal carries a strong diurnal cycle; a "
                         "1-day mean removes it, a 30-day mean leaves only the seasonal march.")
    ap.add_argument("--equator-band", type=float, default=0.5,
                    help="exclude |lat| < this many degrees from the NH/SH means; the o96 grid "
                         "rows at +/-0.4675 deg straddle the equator and belong to neither "
                         "hemisphere (0 = keep them). The global mean always keeps all points.")
    ap.add_argument("--label", action="append", default=None, metavar="RUN=DISPLAY",
                    help="rename a series in the legend, e.g. "
                         "--label sst_exp3_1y_0K=SST-forcing_50_days_0K. Repeatable; only the "
                         "plot is affected, the CSV keeps the real run ids.")
    ap.add_argument("--from-csv", default=None,
                    help="re-plot from a previously written CSV instead of reading the zips "
                         "again (restyling is free; the load is what costs minutes)")
    args = ap.parse_args()

    if args.from_csv:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        times, series = _read_csv(args.from_csv)
        stem = Path(args.from_csv).stem
        region = next((r for r in REGIONS if r != "global" and stem.endswith(f"_{r}")), "global")
        suffix = "" if region == "global" else f"_{region}"
        if args.max_frames:  # --max-frames also truncates a re-plot, e.g. to the first 50 days
            times = times[: args.max_frames]
            series = {k: v[: args.max_frames] for k, v in series.items()}
        print(f"re-plotting {len(series)} series x {len(times)} steps "
              f"from {args.from_csv} (region={region})")
        _plot(Path(args.out_dir), args, times, series, region, suffix)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # {region: {series_name: values}} -- every region is filled from the same single pass
    series = {r: {} for r in REGIONS}
    times = None
    target_done = args.no_target
    for run in args.run:
        print(f"[{run}]")
        src = open_rollout([resolve_zip(run)], args.stream)
        steps = sorted(src)[:: args.stride]
        if args.max_frames:
            steps = steps[: args.max_frames]
        got = mean_series(src, args.field, steps, "prediction",
                          equator_band=args.equator_band)
        if got is None:
            raise ValueError(f"{run} has no prediction group")
        per_region, t = got
        for r in REGIONS:
            series[r][run] = per_region[r]
        if times is None:
            times, ref_steps = t, steps

        # the truth is the same ERA5 field for every run, so read it once, from the first
        if not target_done:
            target_done = True
            got_t = mean_series(src, args.field, steps, "target",
                                equator_band=args.equator_band)
            if got_t is None:
                print("  no target group in this rollout; plotting predictions only")
            else:
                tgt = got_t[0]
                # free-running steps beyond data coverage are written as zeros, not truth
                if np.allclose(np.nan_to_num(tgt["global"]), 0.0):
                    print("  target group is all zeros (free-running beyond coverage); skipping")
                else:
                    for r in REGIONS:
                        series[r]["target (ERA5)"] = tgt[r]

    for region in REGIONS:
        reg_series = series[region]
        suffix = "" if region == "global" else f"_{region}"
        csv = out_dir / f"{args.field}_global_mean_series{suffix}.csv"
        names = list(reg_series)
        with csv.open("w") as fh:
            fh.write("step,time," + ",".join(n.replace(",", "") for n in names) + "\n")
            for i, s in enumerate(ref_steps):
                fh.write(f"{s},{str(times[i])[:19]},"
                         + ",".join(f"{reg_series[n][i]:.5f}" for n in names) + "\n")
        print(f"wrote {csv}")

        print(f"\n[{region}] {'series':<24}{'mean':>10}{'first200':>10}{'last200':>10}{'drift':>10}")
        for n in names:
            v = reg_series[n]
            early, late = float(np.nanmean(v[:200])), float(np.nanmean(v[-200:]))
            print(f"{'':<9}{n:<24}{np.nanmean(v):>10.3f}{early:>10.3f}"
                  f"{late:>10.3f}{late - early:>+10.3f}")

        _plot(out_dir, args, times, reg_series, region, suffix)


def _running_mean(v, win):
    """Centred running mean that keeps the array length (edges use the shorter window)."""
    if win <= 1:
        return v
    kernel = np.ones(win) / win
    padded = np.pad(v, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _apply_labels(args, series):
    """Rename series for display; returns (series, base_name) with the mapping applied."""
    mapping = {}
    for spec in args.label or []:
        if "=" not in spec:
            raise ValueError(f"--label expects RUN=DISPLAY, got {spec!r}")
        old, new = spec.split("=", 1)
        mapping[old] = new
    unknown = [k for k in mapping if k not in series]
    if unknown:
        raise ValueError(f"--label names series not present: {unknown}; have {list(series)}")
    renamed = {mapping.get(k, k): v for k, v in series.items()}
    return renamed, mapping.get(args.run[0], args.run[0])


_REGION_TITLE = {"global": "global", "nh": "northern-hemisphere", "sh": "southern-hemisphere"}


def _plot(out_dir, args, times, series, region="global", suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    series, base_name = _apply_labels(args, series)
    t = times.astype("datetime64[s]").astype("O")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=140, sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})

    win = max(1, int(round(args.smooth_days * 4)))  # 6-hourly steps per day
    for name, v in series.items():
        is_target = name.startswith("target")
        color = "k" if is_target else None
        if win > 1:
            # raw kept as a faint trace so the smoothing is visibly honest about the spread
            line, = ax.plot(t, v, lw=0.5, alpha=0.25, color=color)
            sm = _running_mean(v, win)
            ax.plot(t, sm, lw=1.8 if is_target else 1.3, color=line.get_color(), zorder=5,
                    label=f"{name}  ({np.nanmean(v):.2f} K)")
        else:
            style = dict(lw=1.2, color="k", zorder=5) if is_target else dict(lw=0.8)
            ax.plot(t, v, label=f"{name}  ({np.nanmean(v):.2f} K)", **style)

    ax.set_ylabel(f"{_REGION_TITLE[region]}-mean {args.field} (K)")
    smooth_note = f"  ({args.smooth_days:g}-day running mean; faint = raw 6-hourly)" if win > 1 else ""
    ax.set_title(f"{_REGION_TITLE[region].capitalize()}-mean {args.field} "
                 f"through the free-running rollout{smooth_note}")
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.25)

    # lower panel: same series minus the first run, which makes the spread readable when the
    # seasonal cycle (tens of K) dwarfs the differences between runs (order 1 K)
    base = series[base_name]
    for name, v in series.items():
        if name == base_name:
            continue
        is_target = name.startswith("target")
        color = "k" if is_target else None
        d = v - base
        if win > 1:
            line, = ax2.plot(t, d, lw=0.5, alpha=0.25, color=color)
            ax2.plot(t, _running_mean(d, win), lw=1.5, color=line.get_color(), zorder=5,
                     label=f"{name} - {base_name}  ({np.nanmean(d):+.3f} K)")
        else:
            style = dict(lw=1.4, color="k", zorder=5) if is_target else dict(lw=1.0)
            ax2.plot(t, d, label=f"{name} - {base_name}  ({np.nanmean(d):+.3f} K)", **style)
    ax2.axhline(0, color="0.4", lw=0.8, ls=":")
    ax2.set_ylabel(f"difference vs\n{base_name} (K)")
    ax2.set_xlabel("valid time")
    ax2.legend(loc="best", fontsize=8, ncol=2)
    ax2.grid(alpha=0.25)
    # a 2-year 6-hourly series needs explicit locators; the default picks ~5 ticks and the
    # axis then reads as a single month
    span_days = (t[-1] - t[0]).total_seconds() / 86400.0
    major = 1 if span_days <= 200 else (2 if span_days <= 800 else 6)
    for a in (ax, ax2):
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=major))
        a.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.grid(which="minor", alpha=0.12)
    ax2.set_xlim(t[0], t[-1])
    fig.autofmt_xdate(rotation=45)

    fig.tight_layout()
    png = out_dir / f"{args.field}_global_mean_series{suffix}.png"
    fig.savefig(png)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
