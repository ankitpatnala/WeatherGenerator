#!/usr/bin/env python
"""
Shared reader for WeatherGenerator validation rollout zarrs (ZipStore).

Extracted so the animation and the run-comparison diagnostics load rollouts through exactly
one code path -- the canonical-ordering step below is subtle and must not drift between them.

Layout read here:
  0/<stream>/<step>/{prediction,target}/{data (pts,chan[,ens]), coords (pts,2)=[lat,lon], times}
"""

import numpy as np


def open_rollout(zips, stream="ERA5"):
    """
    Open one or more validation zips as a single {step: group} map.

    Steps present in more than one zip are taken from the LAST zip given, so a corrected
    re-run can be layered over an earlier one.
    """
    import zarr
    import zarr.storage as zs

    src = {}
    for z in zips:
        grp = zarr.open_group(store=zs.ZipStore(z, mode="r"), mode="r")[f"0/{stream}"]
        ks = sorted(int(k) for k in grp.keys())
        for k in ks:
            src[k] = grp
        print(f"  {z}: steps {ks[0]}..{ks[-1]} ({len(ks)})", flush=True)
    return src


def canonical_order(coords):
    """
    Index that sorts grid points into one canonical (lat, lon) ordering.

    The point ORDER is not constant across steps: free-running steps beyond the data emit the
    full grid in raw order, while data-driven steps use the tokenized order. Pairing values
    from different steps (or different runs) without this renders as fine-scale striping while
    leaving the data itself untouched, so every read is mapped through it.
    """
    return np.lexsort((np.round(coords[:, 1], 3), np.round(coords[:, 0], 3)))


def load_field(src, field, steps, group="prediction"):
    """
    Load one channel over `steps` into a (num_steps, num_points) array in canonical order.

    Returns (values, lat, lon, times).
    """
    root = src[steps[0]]
    # channel index is resolved against this run's own channel list, never assumed shared
    ch = list(root[f"{steps[0]}/{group}"].attrs["channels"]).index(field)
    has_ens = root[f"{steps[0]}/{group}/data"].ndim == 3

    coords = np.asarray(root[f"{steps[0]}/{group}/coords"])
    key0 = canonical_order(coords)
    coords = coords[key0]
    lat, lon = coords[:, 0].astype(np.float32), coords[:, 1].astype(np.float32)

    key_cache: dict[bytes, np.ndarray] = {}

    def step_key(c):
        h = np.asarray(c, dtype=np.float32).tobytes()[:4096]
        k = key_cache.get(h)
        if k is None:
            k = canonical_order(c)
            key_cache[h] = k
        return k

    n = len(steps)
    out = np.empty((n, len(lat)), dtype=np.float32)
    times = np.empty(n, dtype="datetime64[ns]")
    for i, s in enumerate(steps):
        nd = src[s][f"{s}/{group}"]
        d = nd["data"]
        v = np.asarray(d[:, ch, 0] if has_ens else d[:, ch])
        out[i] = v[step_key(np.asarray(nd["coords"]))]
        times[i] = np.asarray(nd["times"][0])
        if i % 500 == 0:
            print(f"    loaded {i}/{n}", flush=True)
    return out, lat, lon, times


def resolve_zip(run_id, results="results"):
    """Run name -> its validation zip path."""
    return f"{results}/{run_id}/validation_chkpt00000_rank0000.zip"
