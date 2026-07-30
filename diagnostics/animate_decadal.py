#!/usr/bin/env python
"""
Animate a decadal free-running rollout field to mp4, rendering each frame through the
evaluation package's own map plotter (Plotter.scatter_plot) so frames are identical to eval
score maps (global Robinson projection, coolwarm cmap, coastlines, dashed gridlines,
horizontal colorbar). Frame rendering is parallelised across CPU cores (joblib).

A single FIXED colour range = min/max of the field over the whole run (mirrors the eval
package's _compute_ranges), so drift/collapse is visible; 6-hourly frames resolve the diurnal
cycle.

Reads a WeatherGenerator validation zarr (ZipStore):
  0/<stream>/<step>/prediction/{data (pts,chan,ens), coords (pts,2)=[lat,lon], times}

Usage:
  python diagnostics/animate_decadal.py --field 2t --workers 48
  # -> results/decadal_2023_2026/decadal_2t_animation.mp4
"""

import argparse
import glob
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _render_frames(npy_path, lat, lon, idxs, times_list, steps_list, field, region,
                   vmin, vmax, cmap, markersize, dpi, figsize, stream, frame_dir):
    """Worker: render a subset of frames (by global index) to PNGs via the eval Plotter."""
    import matplotlib
    matplotlib.use("Agg")
    import xarray as xr

    from weathergen.evaluate.plotting.plotter import Plotter

    F = np.load(npy_path, mmap_mode="r")
    cfg = {
        "image_format": "png", "animation_format": "mp4", "dpi_val": dpi,
        "fig_size": tuple(figsize) if figsize else None, "fps": 8, "regions": [region],
    }
    plotter = Plotter(cfg, Path(frame_dir), stream=stream)
    try:
        plotter.update_data_selection({"sample": None, "stream": stream, "forecast_step": 0})
    except Exception:
        pass
    mk = {"vmin": vmin, "vmax": vmax}
    if markersize is not None:
        mk["marker_size"] = markersize
    if cmap is not None:
        mk["colormap"] = cmap

    out = []
    for k, i in enumerate(idxs):
        da = xr.DataArray(
            np.asarray(F[i]), dims=["ipoint"],
            coords={"lat": ("ipoint", lat), "lon": ("ipoint", lon)},
        )
        title = f"{field}  |  {str(times_list[k])[:16]}  (step {steps_list[k]})"
        name = plotter.scatter_plot(
            da, Path(frame_dir), varname=field, regionname=region,
            tag=f"f{i:05d}", map_kwargs=dict(mk), title=title,
        )
        out.append((i, str(Path(frame_dir) / f"{name}.png")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="results/decadal_2023_2026/validation_chkpt00000_rank0000.zip")
    ap.add_argument("--field", default="2t")
    ap.add_argument("--stream", default="ERA5")
    ap.add_argument("--region", default="global")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth step (1 = every 6h)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    ap.add_argument("--markersize", type=float, default=None)
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--figsize", type=float, nargs=2, default=None)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    out = args.out or args.zip.rsplit("/", 1)[0] + f"/decadal_{args.field}_animation.mp4"

    import zarr
    import zarr.storage as zs

    g = zarr.open_group(store=zs.ZipStore(args.zip, mode="r"), mode="r")
    era5 = g[f"0/{args.stream}"]
    steps = sorted(int(k) for k in era5.keys())[:: args.stride]
    if args.max_frames:
        steps = steps[: args.max_frames]
    ch = list(era5[f"{steps[0]}/prediction"].attrs["channels"]).index(args.field)
    coords = np.asarray(era5[f"{steps[0]}/prediction/coords"])
    lat, lon = coords[:, 0].astype(np.float32), coords[:, 1].astype(np.float32)
    n = len(steps)
    print(f"{n} frames, field={args.field} (chan {ch}), {len(lat)} pts, {args.workers} workers -> {out}")

    F = np.empty((n, len(lat)), dtype=np.float32)
    times = np.empty(n, dtype="datetime64[ns]")
    for i, s in enumerate(steps):
        pr = era5[f"{s}/prediction"]
        F[i] = np.asarray(pr["data"][:, ch, 0])
        times[i] = np.asarray(pr["times"][0])
        if i % 500 == 0:
            print(f"  loaded {i}/{n}")

    finite = F[np.isfinite(F)]
    vmin = args.vmin if args.vmin is not None else float(finite.min())
    vmax = args.vmax if args.vmax is not None else float(finite.max())
    print(f"fixed colour range (min/max over run): [{vmin:.3g}, {vmax:.3g}]")

    from joblib import Parallel, delayed
    import imageio.v2 as imageio

    frame_dir = Path(tempfile.mkdtemp(prefix="decadal_frames_"))
    npy_path = str(frame_dir / "_field.npy")
    np.save(npy_path, F)
    times_np = np.array([str(t) for t in times])

    try:
        chunks = [c for c in np.array_split(np.arange(n), args.workers) if len(c)]
        print(f"rendering {n} frames across {len(chunks)} workers ...")
        results = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
            delayed(_render_frames)(
                npy_path, lat, lon, list(c),
                [times_np[i] for i in c], [steps[i] for i in c],
                args.field, args.region, vmin, vmax, args.cmap, args.markersize,
                args.dpi, args.figsize, args.stream, str(frame_dir),
            )
            for c in chunks
        )
        idx_path = sorted((i, p) for sub in results for (i, p) in sub)
        frame_paths = [p for _, p in idx_path]
        print(f"rendered {len(frame_paths)} frames; assembling mp4 ...")

        frames = [imageio.imread(p) for p in frame_paths]
        H = max(f.shape[0] for f in frames); W = max(f.shape[1] for f in frames)
        H += H % 2; W += W % 2

        def _pad(f):
            f = f[..., :3] if f.ndim == 3 and f.shape[2] == 4 else f
            if f.ndim == 2:
                f = np.stack([f] * 3, axis=-1)
            canvas = np.full((H, W, 3), 255, dtype=np.uint8)
            h, w = f.shape[:2]; canvas[:h, :w] = f
            return canvas

        writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=None)
        for f in frames:
            writer.append_data(_pad(f))
        writer.close()
        print(f"saved {out}  ({len(frames)} frames, {W}x{H})")
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
