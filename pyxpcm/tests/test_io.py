#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 2019-03-26
__author__ = 'gmaze@ifremer.fr'

import os
import pyxpcm
from pyxpcm.models import pcm
import pytest
import xarray as xr

def test_data_loader():
    """Test dummy dataset loader"""
    for d in ['dummy', 'argo', 'isas_snapshot', 'isas_series', 'orsi']:
        ds = pyxpcm.tutorial.open_dataset(d).load()
        assert isinstance(ds, xr.Dataset) == True
        print("Load", d, "OK")

def test_saveload_prediction():
    """Test PCM save to netcdf"""
    ds = pyxpcm.tutorial.open_dataset('dummy').load(Np=50, Nz=20)
    pcm_features = {'TEMP': ds['depth'], 'PSAL': ds['depth']}

    # Determine backends to test:
    backends = list()
    try:
        import sklearn
        backends.append('sklearn')
    except ModuleNotFoundError:
        pass
    try:
        import dask_ml
        backends.append('dask_ml')
    except ModuleNotFoundError:
        pass

    # Create a model, fit, predict, save, load, predict
    file = '.pyxpcm_dummy_file.nc'
    for backend in backends:
        for scaling in [0, 1, 2]:
            for reduction in [0, 1]:
                M = pcm(K=3, features=pcm_features, scaling=scaling, reduction=reduction, backend=backend)
                M.fit(ds)
                M.to_netcdf(file, mode='w')
                label_ref = M.predict(ds, inplace=False)
                M_loaded = pyxpcm.load_netcdf(file)
                label_new = M_loaded.predict(ds, inplace=False)
                assert label_ref.equals(label_new) == True, "Netcdf I/O not producing similar results"

    # Delete file at the end of the test:
    os.remove(file)