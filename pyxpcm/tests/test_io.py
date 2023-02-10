#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 2019-03-26
__author__ = 'gmaze@ifremer.fr'

import os
import pytest
import pyxpcm
from pyxpcm.models import pcm
import xarray as xr
from utils import backends, backends_ids

@pytest.mark.parametrize("d", ['dummy', 'argo', 'isas_snapshot', 'isas_series', 'orsi'], indirect=False)
def test_data_loader(d):
    """Test dummy dataset loader"""
    ds = pyxpcm.tutorial.open_dataset(d).load()
    assert isinstance(ds, xr.Dataset) == True


class Test_saveload_prediction:
    """Test PCM save to netcdf"""
    ds = pyxpcm.tutorial.open_dataset('dummy').load(Np=50, Nz=20)
    pcm_features = {'TEMP': ds['depth'], 'PSAL': ds['depth']}

    scaling = [0, 1, 2]
    scaling_ids = ["scaling=%i" % s for s in scaling]

    reduction = [0, 1]
    reduction_ids = ["reduction=%i" % s for s in reduction]

    # Create a model, fit, predict, save, load, predict
    file = '.pyxpcm_dummy_file.nc'

    @pytest.mark.parametrize("scaling", scaling, indirect=False, ids=scaling_ids)
    @pytest.mark.parametrize("reduction", reduction, indirect=False, ids=reduction_ids)
    @pytest.mark.parametrize("backend", backends, indirect=False, ids=backends_ids)
    def test(self, backend, scaling, reduction):
        M = pcm(K=3, features=self.pcm_features, scaling=scaling, reduction=reduction, backend=backend)
        M.fit(self.ds)
        M.to_netcdf(self.file, mode='w')
        label_ref = M.predict(self.ds, inplace=False)
        M_loaded = pyxpcm.load_netcdf(self.file)
        label_new = M_loaded.predict(self.ds, inplace=False)
        assert label_ref.equals(label_new) == True, "Netcdf I/O not producing similar results"

        # Delete file at the end of the test:
        os.remove(self.file)