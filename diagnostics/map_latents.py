#!/usr/bin/env python
"""
Spatially map and compare latent sets A / B / C.

  A  forecast free-run (post-FE)         -- iterated FE from one initial condition
  B  encoder 1-step    (post-FE)         -- FE(encode(real(t))), truth-anchored
  C  encoder pre-FE    (assimilation)    -- encode(real(t)), the FE's input

compare_latents.py reduces each step to a single number; this keeps the healpix cell axis so
you can see WHERE the sets differ. Latents are (cells, dim) at healpix level `--level`
(nested order), so cell index -> lat/lon is a direct healpix lookup.

Per-set maps (time-averaged over the steps read):
  norm        mean_t ||z_t[c]||
  cos_prev    mean_t cos(z_t[c], z_{t-1}[c])      how much each cell evolves per step

Cross-set maps for each --pair X Y (compared at equal step index = equal valid time):
  cos_xy      mean_t cos(zX_t[c], zY_t[c])        directional agreement
  norm_ratio  mean_t ||zX_t[c]|| / ||zY_t[c]||    amplitude ratio

Example
-------
python diagnostics/map_latents.py \
    --series results/oq9o0t86/latents/setA A \
    --series results/oq9o0t86/latents/setB B \
    --series results/oq9o0t86/latents/setC C \
    --pair A C --pair A B --max-steps 200
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np


def iter_latents(d, max_steps=None, num_aux=0):
    """Yield (step, z[cells, dim]) from the chunked z_*.npy dumps, in step order."""
    n = 0
    for f in sorted(glob.glob(os.path.join(d, "z_*.npy"))):
        arr = np.load(f, mmap_mode="r")
        start = int(os.path.basename(f).split("_")[1].split(".")[0])
        block = arr if arr.ndim == 3 else arr[None]
        for j in range(block.shape[0]):
            z = np.asarray(block[j], dtype=np.float32)
            yield start + j, z[num_aux:] if num_aux else z
            n += 1
            if max_steps and n >= max_steps:
                return


def per_cell_stats(d, max_steps, num_aux):
    """Time-averaged per-cell norm and cos-to-previous."""
    s_norm = s_cos = None
    cnt_n = cnt_c = 0
    prev = None
    for _t, z in iter_latents(d, max_steps, num_aux):
        nrm = np.linalg.norm(z, axis=-1)
        s_norm = nrm if s_norm is None else s_norm + nrm
        cnt_n += 1
        if prev is not None:
            num = (z * prev).sum(-1)
            den = nrm * np.linalg.norm(prev, axis=-1) + 1e-12
            c = num / den
            s_cos = c if s_cos is None else s_cos + c
            cnt_c += 1
        prev = z
    return s_norm / max(cnt_n, 1), (s_cos / cnt_c if cnt_c else None), cnt_n


def cross_stats(d1, d2, max_steps, num_aux):
    """Time-averaged per-cell cosine and norm ratio between two sets at equal step index."""
    g1, g2 = iter_latents(d1, max_steps, num_aux), iter_latents(d2, max_steps, num_aux)
    s_cos = s_ratio = None
    n = 0
    for (_t1, a), (_t2, b) in zip(g1, g2, strict=False):
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        c = (a * b).sum(-1) / (na * nb + 1e-12)
        r = na / (nb + 1e-12)
        s_cos = c if s_cos is None else s_cos + c
        s_ratio = r if s_ratio is None else s_ratio + r
        n += 1
    return s_cos / max(n, 1), s_ratio / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs=2, action="append", metavar=("DIR", "LABEL"), required=True)
    ap.add_argument("--pair", nargs=2, action="append", metavar=("X", "Y"), default=[])
    ap.add_argument("--level", type=int, default=5, help="healpix level of the latent grid")
    ap.add_argument("--num-aux", type=int, default=0, help="leading aux tokens to drop")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--stream", default="ERA5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import astropy_healpix as hp
    import astropy.units as u
    import xarray as xr

    from weathergen.evaluate.plotting.plotter import Plotter

    sets = {lab: d for d, lab in args.series}
    out = Path(args.out or (Path(list(sets.values())[0]).parent / "latent_maps"))
    out.mkdir(parents=True, exist_ok=True)

    nside = 2 ** args.level
    ncells = 12 * nside ** 2
    lon, lat = hp.healpix_to_lonlat(np.arange(ncells), nside, order="nested")
    lat = lat.to(u.deg).value.astype(np.float32)
    lon = ((lon.to(u.deg).value + 180.0) % 360.0 - 180.0).astype(np.float32)

    cfg = {"image_format": "png", "animation_format": "mp4", "dpi_val": 130,
           "fig_size": None, "fps": 8, "regions": ["global"]}
    plotter = Plotter(cfg, out, stream=args.stream)
    try:
        plotter.update_data_selection({"sample": None, "stream": args.stream, "forecast_step": 0})
    except Exception:
        pass

    def draw(vals, tag, title, vmin=None, vmax=None, cmap=None):
        if vals is None:
            return
        if len(vals) != ncells:
            print(f"  [skip {tag}] {len(vals)} cells != {ncells} for level {args.level}")
            return
        mk = {"vmin": float(np.nanmin(vals)) if vmin is None else vmin,
              "vmax": float(np.nanmax(vals)) if vmax is None else vmax}
        if cmap:
            mk["colormap"] = cmap
        da = xr.DataArray(vals.astype(np.float32), dims=["ipoint"],
                          coords={"lat": ("ipoint", lat), "lon": ("ipoint", lon)})
        plotter.scatter_plot(da, out, varname="latent", regionname="global",
                             tag=tag, map_kwargs=mk, title=title)
        print(f"  {tag}: mean={np.nanmean(vals):.4f} min={np.nanmin(vals):.4f} "
              f"max={np.nanmax(vals):.4f}")

    for lab, d in sets.items():
        print(f"\n=== {lab}  ({d}) ===")
        nrm, cos, n = per_cell_stats(d, args.max_steps, args.num_aux)
        print(f"  {n} steps read")
        draw(nrm, f"norm_{lab}", f"latent per-cell norm  [{lab}]  (mean of {n} steps)")
        draw(cos, f"cosprev_{lab}",
             f"latent cos to previous step  [{lab}]  (mean of {n} steps)", 0.0, 1.0)

    for x, y in args.pair:
        if x not in sets or y not in sets:
            print(f"[pair {x},{y}] unknown label"); continue
        print(f"\n=== {x} vs {y} ===")
        c, r, n = cross_stats(sets[x], sets[y], args.max_steps, args.num_aux)
        print(f"  {n} step pairs")
        draw(c, f"cos_{x}_{y}", f"per-cell cos({x}, {y})  (mean of {n} steps)", 0.0, 1.0)
        draw(r, f"normratio_{x}_{y}", f"per-cell ||{x}|| / ||{y}||  (mean of {n} steps)")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
                                    
