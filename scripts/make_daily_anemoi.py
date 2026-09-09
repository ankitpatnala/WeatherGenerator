# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Bake an anemoi `rolling_average` + `frequency` pipeline into a new zarr on disk.

Reading

    anemoi_config:
      dataset: <hourly>.zarr
      rolling_average: [-24, 0, "frequency"]
    frequency: 24:00:00

costs 25 full-grid chunk reads (~455 MB decompressed) per daily sample, because the average is
recomputed on every __getitem__ and only every 24th window is kept. This script computes the
same daily series once and writes it as a standalone anemoi dataset, after which the stream
config drops `rolling_average` and reads one chunk per sample.

The output is a drop-in replacement: same variables/order, same grid, and the *hourly*
normalisation statistics copied verbatim, so an existing checkpoint sees identical normalised
inputs. Data values match `np.nanmean` of the same 25 hourly steps bit-for-bit.

Usage
-----
    # 1. create the store + metadata (fast, no data)
    python scripts/make_daily_anemoi.py init  --src <hourly>.zarr --dst <daily>.zarr

    # 2. fill the data, sharded (run as a slurm array over --shard 0..N-1)
    python scripts/make_daily_anemoi.py fill  --dst <daily>.zarr --shard $I --nshards $N

    # 3. check against the on-the-fly pipeline
    python scripts/make_daily_anemoi.py verify --dst <daily>.zarr --num-samples 8
"""

import argparse
import datetime
import logging
import sys
import uuid
from pathlib import Path

import numcodecs
import numpy as np
import zarr

_logger = logging.getLogger("make_daily_anemoi")

# arrays that are per-variable metadata and carry over unchanged (see note on statistics below)
_PER_VARIABLE_PREFIXES = (
    "mean",
    "stdev",
    "minimum",
    "maximum",
    "count",
    "sums",
    "squares",
    "has_nans",
    "statistics_tendencies_",
)


def _src_path(dst: zarr.Group) -> str:
    return dst.attrs["derived_from"]["dataset"]


def _window(dst: zarr.Group) -> tuple[int, int]:
    """(number of hourly steps averaged, output stride in hourly steps)."""
    d = dst.attrs["derived_from"]
    return int(d["window_len"]), int(d["step"])


def num_outputs(n_src: int, window_len: int, step: int) -> int:
    """Number of daily samples, matching len(Subset(RollingAverage(...)))."""
    return (n_src - window_len) // step + 1


def cmd_init(args: argparse.Namespace) -> None:
    src = zarr.open(args.src, mode="r")
    src_data = src["data"]
    src_dates = src["dates"][:]

    i_start = -args.window[0]  # steps before the stamp
    i_end = args.window[1] + 1  # steps at/after the stamp
    if i_start <= 0 or i_end <= 0:
        raise ValueError(f"window must be (negative, non-negative), got {args.window}")
    window_len = i_start + i_end
    step = args.step

    n_out = num_outputs(src_data.shape[0], window_len, step)
    shape = (n_out, *src_data.shape[1:])
    # the stamp of output j is the date at the *end* of its window, as RollingAverage.dates does
    out_dates = src_dates[i_start : i_start + n_out * step : step]

    freq_td = (out_dates[1] - out_dates[0]).astype("timedelta64[s]").astype(int)
    if freq_td % 3600:
        raise ValueError(f"output frequency {freq_td}s is not a whole number of hours")
    frequency = f"{freq_td // 3600}h"

    _logger.info(
        f"{src_data.shape[0]} source steps -> {n_out} outputs "
        f"({window_len}-step mean, stride {step}, frequency {frequency})"
    )
    _logger.info(f"dates {out_dates[0]} .. {out_dates[-1]}")

    dst = zarr.open_group(args.dst, mode="w", zarr_format=2)
    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)

    dst.create_array(
        "data",
        shape=shape,
        chunks=src_data.chunks,  # one full timestep per chunk: shards write disjoint chunks
        dtype=src_data.dtype,
        fill_value=np.nan,
        compressors=[compressor],
    )
    for name in ("dates", "latitudes", "longitudes"):
        values = out_dates if name == "dates" else src[name][:]
        arr = dst.create_array(
            name, shape=values.shape, chunks=values.shape, dtype=values.dtype, compressors=None
        )
        arr[:] = values

    # Statistics are copied from the hourly source rather than recomputed. Averaging removes
    # sub-daily variance, so self-computed stdevs would be smaller and every existing checkpoint
    # would see differently-scaled inputs. Copying keeps this dataset numerically identical to
    # the on-the-fly pipeline, which also forwards the hourly statistics.
    for name, arr in src.arrays():
        if name.startswith(_PER_VARIABLE_PREFIXES):
            values = arr[:]
            out = dst.create_array(
                name, shape=values.shape, chunks=values.shape, dtype=values.dtype, compressors=None
            )
            out[:] = values

    attrs = dict(src.attrs)
    attrs["shape"] = list(shape)
    attrs["chunks"] = list(src_data.chunks)
    attrs["frequency"] = frequency
    attrs["start_date"] = str(out_dates[0].astype("datetime64[s]")).replace(" ", "T")
    attrs["end_date"] = str(out_dates[-1].astype("datetime64[s]")).replace(" ", "T")
    attrs["uuid"] = str(uuid.uuid4())
    attrs["description"] = (
        f"{attrs.get('description', '')} "
        f"[{window_len}-step rolling mean, subsampled to {frequency}]"
    ).strip()
    attrs["derived_from"] = {
        "dataset": str(Path(args.src).resolve()),
        "uuid": src.attrs.get("uuid"),
        "rolling_average": list(args.window) + ["frequency"],
        "window_len": window_len,
        "step": step,
        "statistics": "copied from source (hourly), not recomputed",
    }
    attrs["history"] = list(attrs.get("history", [])) + [
        {
            "action": "rolling_average_baked",
            "timestamp": datetime.datetime.now().isoformat(),
            "source": str(Path(args.src).resolve()),
            "window": list(args.window),
            "step": step,
        }
    ]
    dst.attrs.update(attrs)

    _logger.info(f"initialised {args.dst}: data{shape} {src_data.dtype}")
    _logger.info(f"now run `fill` for shards 0..N-1 (uncompressed size {_gb(shape)} GB)")


def _gb(shape: tuple[int, ...]) -> float:
    return float(np.prod(shape)) * 4 / 1024**3


def cmd_fill(args: argparse.Namespace) -> None:
    dst = zarr.open_group(args.dst, mode="r+")
    src = zarr.open(_src_path(dst), mode="r")
    window_len, step = _window(dst)

    src_data, dst_data = src["data"], dst["data"]
    n_out = dst_data.shape[0]

    # contiguous shards: consecutive windows overlap by (window_len - step) steps, so a shard
    # reads each hourly step it needs exactly once by carrying that overlap across iterations
    edges = np.linspace(0, n_out, args.nshards + 1).round().astype(int)
    j0, j1 = int(edges[args.shard]), int(edges[args.shard + 1])
    if j0 >= j1:
        _logger.info(f"shard {args.shard}: nothing to do")
        return

    overlap = window_len - step
    if overlap < 0:
        raise ValueError(f"step {step} exceeds window {window_len}; windows would skip data")

    _logger.info(f"shard {args.shard}/{args.nshards}: outputs [{j0}, {j1}) of {n_out}")
    start = datetime.datetime.now()

    carry = None  # the trailing `overlap` hourly steps of the previous window
    for j in range(j0, j1):
        a = j * step
        if carry is None:
            block = src_data[a : a + window_len]
        else:
            block = np.concatenate([carry, src_data[a + overlap : a + window_len]], axis=0)
        assert block.shape[0] == window_len, (j, block.shape)

        # identical to anemoi RollingAverage.__getitem__
        dst_data[j] = np.nanmean(block, axis=0)
        carry = block[step:] if overlap else None

        done = j - j0 + 1
        if done % args.log_every == 0 or j == j1 - 1:
            el = (datetime.datetime.now() - start).total_seconds()
            eta = el / done * (j1 - j0 - done)
            _logger.info(
                f"shard {args.shard}: {done}/{j1 - j0} ({el:.0f}s elapsed, ~{eta:.0f}s left)"
            )

    _logger.info(
        f"shard {args.shard}: done in {(datetime.datetime.now() - start).total_seconds():.0f}s"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    import anemoi.datasets as ad

    dst_g = zarr.open_group(args.dst, mode="r")
    window, step = dst_g.attrs["derived_from"]["rolling_average"], _window(dst_g)[1]
    src_path = _src_path(dst_g)

    ref = ad.open_dataset(
        ad.open_dataset({"dataset": src_path, "rolling_average": window}),
        frequency=f"{step}h",
    )
    new = ad.open_dataset(args.dst)

    assert new.variables == ref.variables, "variable order differs"
    assert len(new) == len(ref), f"length differs: {len(new)} vs {len(ref)}"
    assert (new.dates == ref.dates).all(), "dates differ"
    for k in ("mean", "stdev"):
        assert np.array_equal(new.statistics[k], ref.statistics[k]), f"{k} differs"
    _logger.info(f"metadata OK: {len(new)} steps, {new.dates[0]} .. {new.dates[-1]}")

    rng = np.random.default_rng(0)
    for j in sorted(rng.choice(len(new), size=args.num_samples, replace=False).tolist()):
        a, b = new[j], ref[j]
        if np.array_equal(a, b, equal_nan=True):
            _logger.info(f"  [{j}] {new.dates[j]}  bit-identical")
        else:
            d = np.nanmax(np.abs(a - b))
            _logger.error(f"  [{j}] {new.dates[j]}  MISMATCH, max abs diff {d:g}")
            sys.exit(1)
    _logger.info("all sampled steps bit-identical")


def cmd_status(args: argparse.Namespace) -> None:
    """Report which output steps are still unwritten (an unwritten chunk has no file)."""
    dst = zarr.open_group(args.dst, mode="r")
    n_out = dst["data"].shape[0]
    root = Path(args.dst) / "data"
    missing = [j for j in range(n_out) if not (root / f"{j}.0.0.0").exists()]
    _logger.info(f"{n_out - len(missing)}/{n_out} steps written")
    if missing:
        _logger.info(f"first missing: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    init = sub.add_parser("init", help="create the output store and its metadata")
    init.add_argument("--src", required=True, help="source (hourly) anemoi zarr")
    init.add_argument("--dst", required=True, help="output zarr to create")
    init.add_argument(
        "--window",
        type=int,
        nargs=2,
        default=[-24, 0],
        metavar=("START", "END"),
        help="rolling window in source steps, inclusive both ends, as in the stream config "
        "(default: -24 0, i.e. a 25-step mean ending at the stamp)",
    )
    init.add_argument("--step", type=int, default=24, help="output stride in source steps")
    init.set_defaults(func=cmd_init)

    fill = sub.add_parser("fill", help="compute and write a shard of the data")
    fill.add_argument("--dst", required=True)
    fill.add_argument("--shard", type=int, default=0)
    fill.add_argument("--nshards", type=int, default=1)
    fill.add_argument("--log-every", type=int, default=50)
    fill.set_defaults(func=cmd_fill)

    ver = sub.add_parser("verify", help="compare against the on-the-fly anemoi pipeline")
    ver.add_argument("--dst", required=True)
    ver.add_argument("--num-samples", type=int, default=8)
    ver.set_defaults(func=cmd_verify)

    st = sub.add_parser("status", help="report unwritten steps")
    st.add_argument("--dst", required=True)
    st.set_defaults(func=cmd_status)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
