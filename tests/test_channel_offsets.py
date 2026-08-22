# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the `warmup_offset_k` per-channel offsets (prescribed-warming experiments)."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from weathergen.datasets.data_reader_anemoi import parse_channel_offsets

VARIABLES = ["sst", "ci", "2t", "insolation"]


def test_offset_targets_only_named_channel():
    offsets = parse_channel_offsets({"sst": 2}, VARIABLES, "SST")
    np.testing.assert_allclose(offsets, [2.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("value", [2, 2.0, "2", "2k", " +2 K "])
def test_offset_value_formats(value):
    offsets = parse_channel_offsets({"sst": value}, VARIABLES, "SST")
    assert offsets[0] == pytest.approx(2.0)


def test_offset_from_omegaconf_and_case_insensitive_channel():
    cfg = OmegaConf.create({"warmup_offset_k": {"SST": "-1.5k"}})
    offsets = parse_channel_offsets(cfg.warmup_offset_k, VARIABLES, "SST")
    assert offsets[0] == pytest.approx(-1.5)


@pytest.mark.parametrize("cfg", [None, {}, {"sst": 0}])
def test_no_offset_returns_none(cfg):
    assert parse_channel_offsets(cfg, VARIABLES, "SST") is None


def test_unknown_channel_raises():
    with pytest.raises(ValueError, match="not a variable of the dataset"):
        parse_channel_offsets({"tos": 2}, VARIABLES, "SST")


def test_unparsable_value_raises():
    with pytest.raises(ValueError, match="cannot parse offset"):
        parse_channel_offsets({"sst": "warm"}, VARIABLES, "SST")


def test_offset_broadcasts_over_points_and_preserves_nan():
    offsets = parse_channel_offsets({"sst": 2}, VARIABLES, "SST")
    data = np.array(
        [[271.0, 0.0, 280.0, 0.5], [np.nan, np.nan, 285.0, 0.1]], dtype=np.float32
    )
    shifted = data + offsets

    np.testing.assert_allclose(shifted[0], [273.0, 0.0, 280.0, 0.5])
    assert np.isnan(shifted[1, 0])
    np.testing.assert_allclose(shifted[1, 2:], data[1, 2:])
