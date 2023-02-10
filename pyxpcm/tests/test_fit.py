#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test fitting on several datasets with all possible data/feature structures
#
# Created by gmaze on 27/11/2019
__author__ = 'gmaze@ifremer.fr'

import os
import sys
import xarray as xr
import numpy as np
import pyxpcm
from pyxpcm.models import pcm
import pytest
from sklearn.utils import validation
import warnings

from utils import backends, backends_ids


def get_a_dataset(d):
    # Load and set up the dataset:
    ds = pyxpcm.tutorial.open_dataset(d).load()
    ds['TEMP'].attrs['feature_name'] = 'temperature'
    ds['PSAL'].attrs['feature_name'] = 'salinity'
    if d == 'argo':
        ds = ds.rename({'DEPTH': 'depth'})
        ds['depth'] = xr.DataArray(np.linspace(-10, -1405., len(ds['depth'])),
                                   dims='depth')  # Modify vertical axis for test purposes
        ds['depth'].attrs['axis'] = 'Z'

    # Add single-depth level variables:
    ds['SST'] = ds['TEMP'].isel(depth=0).copy()
    ds['SST'].attrs['feature_name'] = 'sst'
    ds['OMT'] = ds['TEMP'].mean(dim='depth').copy()
    ds['OMT'].attrs['feature_name'] = 'omt'
    # print("Dataset surface depth level: ", ds['depth'].values[0])
    # print(ds)
    return ds


class Test_fit:

    configs = [0, 1, 2, 3, 4, 5]
    configs_ids = ["config=%i" % s for s in configs]

    @pytest.fixture
    def _get_a_dataset(self, request):
        yield get_a_dataset(request.param)

    @pytest.mark.parametrize("config", configs, ids=configs_ids, indirect=False)
    @pytest.mark.parametrize("backend", backends, ids=backends_ids, indirect=False)
    @pytest.mark.parametrize("_get_a_dataset", ['dummy', 'argo', 'isas_snapshot', 'isas_series'], indirect=True)
    def test(self, config, backend, _get_a_dataset):
        ds = _get_a_dataset

        if config == 0:
            # Single feature, vertical axis from dataset
            z = ds['depth'].where(ds['depth'] >= -200, drop=True)
            pcm_features = {'temperature': z}

        elif config == 1:
            # Two features, vertical axis from dataset
            z = ds['depth'].where(ds['depth'] >= -200, drop=True)
            pcm_features = {'temperature': z, 'salinity': z}

        elif config == 2:
            # Single feature, new vertical axis (from -10. to avoid vertical mixing)
            z = np.arange(-10., -500, -10.1)
            pcm_features = {'temperature': z}

        elif config == 3:
            # Two features, new vertical axis (from surface to trigger vertical mixing)
            z = np.arange(0., -500, -10.1)
            pcm_features = {'temperature': z, 'salinity': z}

        elif config == 4:
            # Two features, new vertical axis (from surface to trigger vertical mixing) and a slice feature (single-level)
            z = np.arange(0., -500, -10.1)
            pcm_features = {'temperature': z, 'sst': None}

        elif config == 5:
            # Two slice features:
            pcm_features = {'omt': None, 'sst': None}

        #
        # default_opts = {'K': 3, 'features': pcm_features, 'reduction': 1, 'debug': 0, 'timeit': 0, 'chunk_size': 'auto'}
        default_opts = {'K': 3, 'features': pcm_features} # Default, no options specified

        # print("\n", "=" * 60, "CONFIG %i / %s / %s" % (config, d, backend), list(pcm_features.keys()))
        default_opts['backend'] = backend
        m = pcm(**default_opts)
        m.fit(ds)
        assert m.fitted
