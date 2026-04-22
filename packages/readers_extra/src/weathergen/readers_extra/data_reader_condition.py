# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import warnings
from pathlib import Path
from typing import override

import numpy as np
import astropy_healpix as hp

from weathergen.datasets.data_reader_base import (
    DataReaderTimestep,
    ReaderData,
    TimeWindowHandler,
    TIndex,
    DTRange
)

_logger = logging.getLogger(__name__)


class DataReaderCondition(DataReaderTimestep):
    "Wrapper for forecast condition variables derived from time window metadata"

    def __init__(
        self,
        tw_handler: TimeWindowHandler,
        filename: Path,
        stream_info: dict,
    ) -> None:
        """
        Construct data reader for forecast condition variables.

        Parameters
        ----------
        tw_handler :
            time window handler
        filename :
            unused; kept for interface compatibility (filenames should be empty)
        stream_info :
            information about stream; must include 'transform' and 'variables'

        Returns
        -------
        None
        """
        self.source_idx = []
        self.source_channels = []
        self.target_channels = []
        self.geoinfo_channels = []
        self.target_idx = []
        self.geoinfo_idx = []
        self.target_channel_weights = []
        self.condition_idx = []
        self.healpix_order = stream_info.get("healpix_order", 5) 
        self.num_healpix_cells = hp.nside_to_npix(hp.level_to_nside(self.healpix_order))
        self.transform: str = stream_info.get("transform", "absolute")
        self.variables: list[str] = list(
            stream_info.get("variables", ["start_day", "start_time", "end_day", "end_time"])
        )
        self.normalize = stream_info.get("normalize", False)
        self.num_channels: int = self._compute_num_channels(stream_info)
        self.source_idx = []

        self.filetype = stream_info.get("filetype", None)
        
        self.ds = None
        print(f"Initializing condition data reader with filetype {self.filetype!r}")
        match self.filetype:
            case "anemoi":
                from anemoi import datasets as datasets
                print(f"Opening dataset from {filename}")
                self.ds = datasets.open_dataset(filename)
                self._select_channels(stream_info)
                print(f"Dataset opened with variables: {self.ds.variables}")
                self.per_cell_order = self.accumulate_per_cell()
        
        super().__init__(
            tw_handler,
            stream_info,
            tw_handler.t_start,
            tw_handler.t_end,
            tw_handler.t_window_step,
        )

        self.len = int((tw_handler.t_end - tw_handler.t_start) // tw_handler.t_window_step)

    def _select_channels(self, stream_info: dict) -> list[int]:
        variables = self.ds.variables
        for source_idx, source in enumerate(stream_info.get("source", [])):
            if source in variables:
                self.source_idx.append(source_idx)
                self.source_channels.append(variables.index(source))
            
    def obtain_time_indices(self, dtr) -> np.ndarray:
        dates = self.ds.dates.astype("datetime64[s]")
        return np.where((dates >= dtr.start) & (dates < dtr.end))[0]

    def get_healpix_cell_indices(
    self,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    healpix_order: int,
    nest: bool = True,
) -> np.ndarray:
        """
        Map lat/lon coordinates to HEALPix cell indices.

        Parameters
        ----------
        latitudes  : (S,) geographic latitude  in degrees, range [-90, 90]
        longitudes : (S,) geographic longitude in degrees, range [0, 360]
        healpix_order : HEALPix order k, where nside = 2^k
        nest       : True = NESTED scheme (default), False = RING

        Returns
        -------
        pixel_indices : (S,) int array of HEALPix cell indices
        """
        import astropy_healpix as hp
        import numpy as np

        nside = 2 ** healpix_order

        theta = np.radians(90.0 - latitudes)   # co-latitude [0, π]
        phi   = np.radians(longitudes)          # longitude   [0, 2π]

        return hp.healpy.ang2pix(nside, theta, phi, nest=nest)
    
    
    def accumulate_per_cell(self) -> None:
        latitudes = self.ds.latitudes
        longitudes = self.ds.longitudes

        pixel_indices = self.get_healpix_cell_indices(latitudes, longitudes, self.healpix_order)
        order = np.argsort(pixel_indices, stable=True)
        sorted_cells = pixel_indices[order]

        boundaries = np.searchsorted(sorted_cells, np.arange(self.num_healpix_cells))
        boundaries_end = np.searchsorted(sorted_cells, np.arange(self.num_healpix_cells), side='right')

        # precomputed for vectorized _get_source: one read of all stations then bincount
        self.sorted_order = order
        self.cell_labels = np.repeat(
            np.arange(self.num_healpix_cells), boundaries_end - boundaries
        )

        return [order[boundaries[c]:boundaries_end[c]].tolist() for c in range(self.num_healpix_cells)]
        

    def _compute_num_channels(self, stream_info: dict) -> int:
        if self.transform == "absolute":
            return len(self.variables)
        elif self.transform == "cos_sin":
            return 2 * len(self.variables)
        elif self.transform == "fourier":
            assert "emb_dimension" in stream_info, "Fourier transform requires 'emb_dimension' in stream_info"
            return stream_info.get("emb_dimension")* len(self.variables)
        else:
            raise ValueError(f"Unknown transform: {self.transform!r}")

    @override
    def init_empty(self) -> None:
        super().init_empty()
        self.len = 0

    @override
    def length(self) -> int:
        return self.len

    @override
    def _get(self, idx: TIndex) -> ReaderData:
        scalar_data = self._get_scalar(idx, self.target_channels)
        source_data = self._get_source(idx)
        return scalar_data, source_data

    def _get_scalar(self, idx: TIndex, channels_idx: list[int]) -> ReaderData:
        """
        Compute condition variables for a given time window.

        Parameters
        ----------
        idx : TIndex
            Index of temporal window
        channels_idx : list[int]
            Selection of channels

        Returns
        -------
        ReaderData providing coords, geoinfos, data, datetimes
        """

        dtr = self.time_window_handler.window(idx)
        encoded_conditions = self._encode(dtr, self.variables)
        return encoded_conditions
    
    def _get_source(self, idx: TIndex) -> np.ndarray:
        """
        Get source data for a given time window.

        Parameters
        ----------
        idx : TIndex
            Index of temporal window

        Returns
        -------
        np.ndarray of shape (num_cells, num_source_channels)
        """
        if self.ds is None:
            return np.empty((self.num_healpix_cells, 0), dtype=np.float32)

        dtr = self.time_window_handler.window(idx)
        time_indices = self.obtain_time_indices(dtr)

        # single read for all stations pre-sorted by healpix cell (vs one oindex per cell)
        all_data = self.ds.data.oindex[time_indices, self.source_channels, :, self.sorted_order]

        if self.normalize:
            mean = self.ds.statistics['mean'][self.source_channels]
            stdev = self.ds.statistics['stdev'][self.source_channels]
            all_data = (all_data - mean) / stdev

        # collapse to (num_channels, num_stations) — average over time/ensemble dims if present
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            while all_data.ndim > 2:
                all_data = np.nanmean(all_data, axis=0)

        # vectorized per-cell nanmean via bincount — one pass per channel (C is small)
        num_channels = all_data.shape[0]
        result = np.empty((self.num_healpix_cells, num_channels), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for c in range(num_channels):
                ch = all_data[c]
                nan_mask = np.isnan(ch)
                cell_sums = np.bincount(
                    self.cell_labels, weights=np.where(nan_mask, 0.0, ch), minlength=self.num_healpix_cells
                )
                cell_counts = np.bincount(
                    self.cell_labels, weights=(~nan_mask).astype(np.float32), minlength=self.num_healpix_cells
                )
                result[:, c] = np.where(cell_counts > 0, cell_sums / cell_counts, 0.0)

        return result

    def _encode(self, dtr: DTRange, variables: list[str]) -> np.ndarray:
        """
        Encode start/end datetimes into condition variable values.

        Parameters
        ----------
        start_dt :
            start of time window
        end_dt :
            end of time window

        Returns
        -------
        np.ndarray of shape (num_channels,)
        """

        values: list[float] = []
        # Right now we only support absolute encoding, fourier or sin_cosine is not supported yet
        for var in self.variables:
            if var == "start_day":
                values.append(_day_of_year(dtr.start))
            elif var == "start_time":
                values.append(_hour_of_day(dtr.start))
            elif var == "end_day":  # noqa: PLR200
                values.append(_day_of_year(dtr.end))
            elif var == "end_time":  # noqa: PLR200
                values.append(_hour_of_day(dtr.end))

        return  values


def _day_of_year(dt: np.datetime64) -> float:
    """Return 1-indexed day of year for a numpy datetime64."""
    jan1 = dt.astype("datetime64[Y]").astype("datetime64[D]")
    return float((dt.astype("datetime64[D]") - jan1).astype(int) + 1)


def _hour_of_day(dt: np.datetime64) -> float:
    """Return fractional hour of day [0, 24) for a numpy datetime64."""
    day = dt.astype("datetime64[D]")
    return float((dt - day) / np.timedelta64(1, "h"))
