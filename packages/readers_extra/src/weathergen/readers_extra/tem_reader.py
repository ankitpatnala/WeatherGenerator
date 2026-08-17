# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from pathlib import Path
from typing import override

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    check_reader_data,
)
from weathergen.train.utils import Stage

_logger = logging.getLogger(__name__)


class DataReaderTEM(DataReaderTimestep):
    """
    Data reader for TEM EP flux Zarr datasets.

    Dataset layout: data(time, channels, ensemble, grid_points) on an O96 unstructured grid.
    Statistics (mean/stdev per channel) are loaded from a separate JSON file.
    """

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
        stage: Stage,
    ) -> None:
        self._filename = filename
        self._tw_handler = tw_handler
        self._stream_info = stream_info
        self._stage = stage
        self._initialized = False

        super().__init__(tw_handler, stream_info)

        self.latitudes: NDArray | None = None
        self.longitudes: NDArray | None = None
        self.n_points: int = 0

        self.init_empty()
        self._lazy_init()

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        try:
            ds: xr.Dataset = xr.open_zarr(str(self._filename), consolidated=False, chunks=None)
        except Exception as e:
            name = self._stream_info["name"]
            _logger.error(f"Failed to open {name} at {self._filename}: {e}")
            return

        # ---- Time axis -------------------------------------------------------
        time_coord: NDArray = ds.coords["time"].values
        data_start_time = np.datetime64(time_coord[0])
        data_end_time = np.datetime64(time_coord[-1])

        if self._tw_handler.t_start >= data_end_time or self._tw_handler.t_end <= data_start_time:
            name = self._stream_info["name"]
            _logger.warning(f"{name} is not supported over data loader window. Stream is skipped.")
            return

        period = np.timedelta64(time_coord[1] - time_coord[0]) if len(time_coord) > 1 else np.timedelta64(6, "h")

        # Patch instance attributes that DataReaderTimestep._get_dataset_idxs reads.
        self.data_start_time = data_start_time
        self.data_end_time = data_end_time
        self.period = period

        # ---- Channel names (from coordinate) ---------------------------------
        self.available_vars: list[str] = list(ds.coords["channels"].values)

        # ---- Spatial grid (unstructured O96) ---------------------------------
        self.latitudes = np.clip(ds.coords["latitude"].values.astype(np.float32), -90.0, 90.0)
        self.longitudes = ((ds.coords["longitude"].values.astype(np.float32) + 180.0) % 360.0 - 180.0)
        self.n_points = len(self.latitudes)

        # ---- Channel selection -----------------------------------------------
        source_filter = self._stream_info.get("source")
        source_exclude = self._stream_info.get("source_exclude", [])
        self.source_channels, self.source_idx = self._select_channels(
            self.available_vars, source_filter, source_exclude
        )
        self.source_idx = list(self.source_idx)

        target_filter = self._stream_info.get("target")
        target_exclude = self._stream_info.get("target_exclude", [])
        self.target_channels, self.target_idx = self._select_channels(
            self.available_vars, target_filter, target_exclude
        )
        self.target_idx = list(self.target_idx)

        self.geoinfo_channels = []
        self.geoinfo_idx = np.array([], dtype=np.int64)
        self.mean_geoinfo = np.zeros(0, dtype=np.float32)
        self.stdev_geoinfo = np.ones(0, dtype=np.float32)

        self.target_channel_weights = self.parse_target_channel_weights()

        # ---- Statistics from JSON --------------------------------------------
        self.mean, self.stdev = self._load_statistics()

        # ---- Length ----------------------------------------------------------
        time_mask = (time_coord >= self._tw_handler.t_start) & (time_coord < self._tw_handler.t_end)
        self.len = int(np.sum(time_mask))

        self.ds = ds
        self.properties = {"stream_id": self._stream_info.get("stream_id", 0)}

        ds_name = self._stream_info["name"]
        _logger.info(f"{ds_name}: source channels ({len(self.source_channels)}): {self.source_channels}")
        _logger.info(f"{ds_name}: target channels ({len(self.target_channels)}): {self.target_channels}")
        _logger.info(f"{ds_name}: grid points: {self.n_points}, timesteps in window: {self.len}")

    def _select_channels(
        self,
        available_vars: list[str],
        include_filters: list[str] | None,
        exclude_filters: list[str] | None = None,
    ) -> tuple[list[str], NDArray[np.int64]]:
        if exclude_filters is None:
            exclude_filters = []
        selected_names: list[str] = []
        selected_idxs: list[int] = []
        for i, var in enumerate(available_vars):
            if include_filters is not None:
                if not any(f in var or f == var for f in include_filters):
                    continue
            if any(f in var for f in exclude_filters):
                continue
            selected_names.append(var)
            selected_idxs.append(i)
        return selected_names, np.array(selected_idxs, dtype=np.int64)

    def _load_statistics(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        n = len(self.available_vars)
        mean = np.zeros(n, dtype=np.float32)
        stdev = np.ones(n, dtype=np.float32)

        stats_path = self._stream_info.get("statistics")
        if stats_path is None:
            _logger.warning(f"{self._stream_info['name']}: no statistics path provided; using 0/1 normalization.")
            return mean, stdev

        with open(stats_path) as f:
            stats = json.load(f)

        # JSON format: {"channels": [...], "mean": [...], "stdev": [...]}
        channels_in_stats = stats.get("channels", [])
        means_in_stats = stats.get("mean", [])
        stdevs_in_stats = stats.get("stdev", [])

        stats_lookup = {ch: (m, s) for ch, m, s in zip(channels_in_stats, means_in_stats, stdevs_in_stats)}

        for i, ch in enumerate(self.available_vars):
            if ch in stats_lookup:
                mean[i] = float(stats_lookup[ch][0])
                stdev[i] = float(stats_lookup[ch][1])
            else:
                _logger.warning(f"{self._stream_info['name']}: no statistics for channel {ch}, using 0/1.")

        stdev[stdev <= 1e-5] = 1.0
        return mean, stdev

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.ds = None
        self.len = 0
        self.n_points = 0
        self.available_vars = []

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        self._lazy_init()

        if not channels_idx:
            return ReaderData.empty(num_data_fields=0, num_geo_fields=0)

        (t_idxs, dtr) = self._get_dataset_idxs(idx)

        if self.ds is None or self.len == 0 or len(t_idxs) == 0:
            return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

        time_values = self.ds.coords["time"].values
        n_time = len(time_values)

        data_arrays: list[NDArray] = []
        datetimes_list: list[np.datetime64] = []

        for t_idx in t_idxs:
            if t_idx < 0 or t_idx >= n_time:
                continue

            # ds["data"] shape: (time, channels, ensemble, grid_points)
            # Select all channels, drop ensemble dim → (channels, grid_points)
            raw = self.ds["data"].isel(time=int(t_idx), ensemble=0).values.astype(np.float32)
            # Select requested channels and transpose → (grid_points, len(channels_idx))
            timestep_data = raw[channels_idx].T
            data_arrays.append(timestep_data)
            datetimes_list.extend([np.datetime64(time_values[t_idx])] * self.n_points)

        if not data_arrays:
            return ReaderData.empty(num_data_fields=len(channels_idx), num_geo_fields=0)

        data = np.vstack(data_arrays)  # (n_valid_timesteps * n_points, len(channels_idx))
        n_valid = len(data_arrays)

        coords_single = np.stack([self.latitudes, self.longitudes], axis=1).astype(np.float32)
        coords = np.tile(coords_single, (n_valid, 1))
        geoinfos = np.zeros((len(data), 0), dtype=np.float32)
        datetimes = np.array(datetimes_list, dtype="datetime64[s]")

        rd = ReaderData(
            coords=coords,
            geoinfos=geoinfos,
            data=data,
            datetimes=datetimes,
        )
        check_reader_data(rd, dtr)
        return rd
