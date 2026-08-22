#!/usr/bin/env python
"""
Animate the SAME field from several runs side by side, on ONE shared colour scale.

Each run gets a panel; panels are rendered through the evaluation package's own map plotter
(so they match eval score maps) and stitched horizontally into a single frame. The colour
range is the min/max over ALL runs together unless --vmin/--vmax are given -- without that,
per-panel ranges would rescale each run independently and a genuine warming would be invisible
because every panel would re-normalise it away.

Usage:
  python diagnostics/animate_side_by_side.py --run sst_exp1_1y_0K --run sst_exp1_1y_2K \
      --run sst_exp1_1y_4K --field 2t --workers 64
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_io import load_field, open_rollout, resolve_zip  # noqa: E402


def _render_chunk(npy_paths, labels, lat, lon, idxs, times_list, steps_list, field, region,
                  vmin, vmax, cmap, markersize, dpi, figsize, stream, frame_dir):
    """Worker: render one horizontal strip of panels per frame, for a subset of frames."""
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import xarray as xr

    from weathergen.evaluate.plotting.plotter import Plotter

    fields = [np.load(p, mmap_mode="r") for p in npy_paths]
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
        panels = []
        for j, (F, label) in enumerate(zip(fields, labels, strict=True)):
            da = xr.DataArray(
                np.asarray(F[i]), dims=["ipoint"],
                coords={"lat": ("ipoint", lat), "lon": ("ipoint", lon)},
            )
            title = f"{label}  |  {field}  |  {str(times_list[k])[:16]}  (step {steps_list[k]})"
            name = plotter.scatter_plot(
                da, Path(frame_dir), varname=field, regionname=region,
                tag=f"p{j}_f{i:05d}", map_kwargs=dict(mk), title=title,
            )
            panels.append(Path(frame_dir) / f"{name}.png")

        imgs = [imageio.imread(p) for p in panels]
        imgs = [im[..., :3] if im.ndim == 3 and im.shape[2] == 4 else im for im in imgs]
        imgs = [np.stack([im] * 3, -1) if im.ndim == 2 else im for im in imgs]
        h = max(im.shape[0] for im in imgs)
        w = max(im.shape[1] for im in imgs)
        canvas = np.full((h, w * len(imgs), 3), 255, dtype=np.uint8)
        for j, im in enumerate(imgs):
            canvas[: im.shape[0], j * w : j * w + im.shape[1]] = im
        strip = Path(frame_dir) / f"strip_{i:05d}.png"
        imageio.imwrite(strip, canvas)
        for p in panels:            # the per-panel PNGs are intermediates only
            p.unlink(missing_ok=True)
        out.append((i, str(strip)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="run id; repeatable, panels appear left-to-right in this order")
    ap.add_argument("--field", default="2t")
    ap.add_argument("--stream", default="ERA5")
    ap.add_argument("--source", default="prediction", choices=["prediction", "target"])
    ap.add_argument("--region", default="global")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 8))
    ap.add_argument("--markersize", type=float, default=None)
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--figsize", type=float, nargs=2, default=None)
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = args.out or f"results/comparison/sidebyside_{args.field}_" + "_".join(args.run) + ".mp4"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    # load every run onto the same canonical grid ordering and the same step list
    srcs = {}
    for run in args.run:
        print(f"[{run}]")
        srcs[run] = open_rollout([resolve_zip(run)], args.stream)
    steps = sorted(srcs[args.run[0]])[:: args.stride]
    if args.max_frames:
        steps = steps[: args.max_frames]

    fields, lat, lon, times = {}, None, None, None
    for run in args.run:
        missing = [s for s in steps if s not in srcs[run]]
        if missing:
            raise ValueError(f"{run} is missing {len(missing)} steps (first: {missing[:5]})")
        vals, lat_r, lon_r, times_r = load_field(srcs[run], args.field, steps, args.source)
        if lat is None:
            lat, lon, times = lat_r, lon_r, times_r
        else:
            # panels are compared visually cell-by-cell, so the grids must really match
            if not (np.allclose(lat, lat_r, atol=1e-5) and np.allclose(lon, lon_r, atol=1e-5)):
                raise ValueError(f"{run} is on a different grid than {args.run[0]}")
            bad = int((times != times_r).sum())
            if bad:
                raise ValueError(f"{run} valid times differ from {args.run[0]} at {bad} steps")
        fields[run] = vals

    stack = np.concatenate([f[np.isfinite(f)] for f in fields.values()])
    vmin = args.vmin if args.vmin is not None else float(stack.min())
    vmax = args.vmax if args.vmax is not None else float(stack.max())
    print(f"shared colour range over {len(args.run)} runs: [{vmin:.4g}, {vmax:.4g}]")
    for run, f in fields.items():
        print(f"  {run:<22} mean={float(np.nanmean(f)):+.4f}  "
              f"min={float(np.nanmin(f)):.4g}  max={float(np.nanmax(f)):.4g}")

    n = len(steps)
    keep = args.frames_dir is not None
    frame_dir = Path(args.frames_dir) if keep else Path(tempfile.mkdtemp(prefix="sbs_frames_"))
    frame_dir.mkdir(parents=True, exist_ok=True)
    npy_paths = []
    for run in args.run:
        p = str(frame_dir / f"_field_{run}.npy")
        np.save(p, fields[run])
        npy_paths.append(p)
    del fields
    times_np = np.array([str(t) for t in times])

    from joblib import Parallel, delayed
    import imageio.v2 as imageio

    try:
        chunks = [c for c in np.array_split(np.arange(n), args.workers) if len(c)]
        print(f"rendering {n} frames x {len(args.run)} panels across {len(chunks)} workers ...")
        results = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(
            delayed(_render_chunk)(
                npy_paths, args.run, lat, lon, list(c),
                [times_np[i] for i in c], [steps[i] for i in c],
                args.field, args.region, vmin, vmax, args.cmap, args.markersize,
                args.dpi, args.figsize, args.stream, str(frame_dir),
            )
            for c in chunks
        )
        frame_paths = [p for _, p in sorted((i, p) for sub in results for (i, p) in sub)]
        print(f"rendered {len(frame_paths)} frames; assembling mp4 ...")
        if args.no_video:
            print(f"frames only: {len(frame_paths)} PNGs in {frame_dir}")
            return

        first = imageio.imread(frame_paths[0])
        H, W = first.shape[0] + first.shape[0] % 2, first.shape[1] + first.shape[1] % 2
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=None)
        for p in frame_paths:
            f = imageio.imread(p)
            f = f[..., :3] if f.ndim == 3 and f.shape[2] == 4 else f
            canvas = np.full((H, W, 3), 255, dtype=np.uint8)
            canvas[: f.shape[0], : f.shape[1]] = f
            writer.append_data(canvas)
        writer.close()
        print(f"saved {out}  ({len(frame_paths)} frames, {W}x{H})")
    finally:
        if not keep:
            shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
