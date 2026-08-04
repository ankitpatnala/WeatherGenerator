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
                   vmin, vmax, cmap, markersize, dpi, figsize, stream, frame_dir, tag_src=""):
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
        title = f"{field}{tag_src}  |  {str(times_list[k])[:16]}  (step {steps_list[k]})"
        name = plotter.scatter_plot(
            da, Path(frame_dir), varname=field, regionname=region,
            tag=f"f{i:05d}", map_kwargs=dict(mk), title=title,
        )
        out.append((i, str(Path(frame_dir) / f"{name}.png")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="append", default=None,
                    help="validation zarr; repeatable. Steps present in more than one are\n"
                         "taken from the LAST zip given, so a corrected re-run can be layered\n"
                         "over an earlier one without re-generating the whole rollout.")
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
    ap.add_argument("--frames-dir", default=None,
                    help="write the PNG frames here and keep them (default: a temp dir that is "
                         "deleted after the mp4 is assembled)")
    ap.add_argument("--no-video", action="store_true",
                    help="render frames only; skip mp4 assembly")
    ap.add_argument(
        "--source",
        default="prediction",
        choices=["prediction", "target", "bias"],
        help="field to animate: model output, ERA5 truth, or bias (prediction - target)",
    )
    args = ap.parse_args()
    zips = args.zip or ["results/decadal_2023_Jan_3y_2nd/validation_chkpt00000_rank0000.zip"]
    suffix = "" if args.source == "prediction" else f"_{args.source}"
    out = (
        args.out
        or zips[-1].rsplit("/", 1)[0] + f"/decadal_{args.field}{suffix}_animation.mp4"
    )

    import zarr
    import zarr.storage as zs

    # step -> the zarr group it should be read from; later zips override earlier ones
    src = {}
    for z in zips:
        grp = zarr.open_group(store=zs.ZipStore(z, mode="r"), mode="r")[f"0/{args.stream}"]
        ks = sorted(int(k) for k in grp.keys())
        for k in ks:
            src[k] = grp
        print(f"  {z}: steps {ks[0]}..{ks[-1]} ({len(ks)})")
    steps = sorted(src)[:: args.stride]
    if args.max_frames:
        steps = steps[: args.max_frames]
    era5 = src[steps[0]]
    # prediction is (pts, chan, ens); target is (pts, chan) -- and the two groups carry their
    # own channel lists, so the index is resolved per group rather than assumed shared.
    def _chan(group):
        return list(era5[f"{steps[0]}/{group}"].attrs["channels"]).index(args.field)

    # every zip must describe the same grid/channels or the frames would not be comparable
    for grp in {id(v): v for v in src.values()}.values():
        k = min(int(x) for x in grp.keys())
        assert list(grp[f"{k}/prediction"].attrs["channels"]) == list(
            era5[f"{steps[0]}/prediction"].attrs["channels"]
        ), "channel lists differ between zips"

    ref_group = "target" if args.source == "target" else "prediction"
    coords = np.asarray(era5[f"{steps[0]}/{ref_group}/coords"])

    # The point ORDER is not constant across steps: free-running steps beyond the data emit the
    # full grid in raw order, while data-driven steps use the tokenized order. Reading coords
    # once and reusing them pairs every later frame's values with the wrong locations, which
    # renders as fine-scale striping while leaving the data untouched. So each step's values are
    # mapped into one canonical (lat, lon) ordering before use.
    def _order(c):
        return np.lexsort((np.round(c[:, 1], 3), np.round(c[:, 0], 3)))

    _canon_key = _order(coords)
    coords = coords[_canon_key]
    lat, lon = coords[:, 0].astype(np.float32), coords[:, 1].astype(np.float32)

    _key_cache: dict[bytes, np.ndarray] = {}

    def _step_key(c):
        h = np.asarray(c, dtype=np.float32).tobytes()[:4096]
        k = _key_cache.get(h)
        if k is None:
            k = _order(c)
            _key_cache[h] = k
        return k
    n = len(steps)
    print(f"{n} frames, field={args.field}, source={args.source}, {len(lat)} pts, "
          f"{args.workers} workers -> {out}")

    def _load(group):
        ch = _chan(group)
        has_ens = era5[f"{steps[0]}/{group}/data"].ndim == 3
        arr = np.empty((n, len(lat)), dtype=np.float32)
        tms = np.empty(n, dtype="datetime64[ns]")
        for i, s in enumerate(steps):
            nd = src[s][f"{s}/{group}"]
            d = nd["data"]
            v = np.asarray(d[:, ch, 0] if has_ens else d[:, ch])
            arr[i] = v[_step_key(np.asarray(nd["coords"]))]   # -> canonical order
            tms[i] = np.asarray(nd["times"][0])
            if i % 500 == 0:
                print(f"  [{group}] loaded {i}/{n}", flush=True)
        return arr, tms

    if args.source == "prediction":
        F, times = _load("prediction")
    elif args.source == "target":
        F, times = _load("target")
    else:  # bias = prediction - target
        P, times = _load("prediction")
        T, times_t = _load("target")
        # the two groups are written from the same batch, but subtracting silently across a
        # mismatch would produce a plausible-looking and completely wrong field, so check.
        tc = np.asarray(era5[f"{steps[0]}/target/coords"])
        if not np.allclose(coords, tc, atol=1e-5):
            raise ValueError("prediction and target coords differ; refusing to subtract")
        bad = int((times != times_t).sum())
        if bad:
            raise ValueError(f"prediction/target times differ at {bad}/{n} steps")
        F = P - T

    finite = F[np.isfinite(F)]
    if args.source == "bias":
        # diverging field: symmetric about zero so the colour map reads sign correctly
        m = float(max(abs(finite.min()), abs(finite.max())))
        vmin = args.vmin if args.vmin is not None else -m
        vmax = args.vmax if args.vmax is not None else m
        print(f"symmetric bias colour range: [{vmin:.3g}, {vmax:.3g}]  "
              f"mean bias={float(finite.mean()):+.4g}  rms={float(np.sqrt((finite**2).mean())):.4g}")
    else:
        vmin = args.vmin if args.vmin is not None else float(finite.min())
        vmax = args.vmax if args.vmax is not None else float(finite.max())
        print(f"fixed colour range (min/max over run): [{vmin:.3g}, {vmax:.3g}]")

    from joblib import Parallel, delayed
    import imageio.v2 as imageio

    keep_frames = args.frames_dir is not None
    if keep_frames:
        frame_dir = Path(args.frames_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
    else:
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
                "" if args.source == "prediction" else f"  [{args.source}]",
            )
            for c in chunks
        )
        idx_path = sorted((i, p) for sub in results for (i, p) in sub)
        frame_paths = [p for _, p in idx_path]
        print(f"rendered {len(frame_paths)} frames; assembling mp4 ...")

        if args.no_video:
            print(f"frames only: {len(frame_paths)} PNGs in {frame_dir}")
            return

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
        if not keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

