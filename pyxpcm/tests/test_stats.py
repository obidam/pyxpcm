#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 2019-03-26
__author__ = 'gmaze@ifremer.fr'

from pyxpcm.pcmodel import pcm
from pyxpcm.pcmodel import PCMFeatureError
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from pyxpcm import datasets as pcmdata
from pyxpcm import stats as pcmstats
from pyxpcm import plot as pcmplot
import numpy as np
import xarray as xr
import pytest

def new_m():
    return pcm(K=8, feature_axis=np.arange(-500, 0, 2), feature_name='temperature')

def test_quant():
    """Test PCM stats quant method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Compute Quantiles
    # quant(ds,
    #           of=None,
    #           using='LABEL',
    #           q=[0.05, 0.5, 0.95],
    #           inplace=True,
    #           dim=None,
    #           name='QUANT')

    with pytest.raises(ValueError):
        # This will raise an error because it will look for a variable that is not in the dataset:
        m = new_m()
        dsl = ds.copy()
        m.fit_predict(dsl, feature={'temperature': 'TEMP'}, inplace=True)
        pcmstats.quant(dsl, of='UNKNOWN_VARIABLE')

    with pytest.raises(ValueError):
        # This will raise an error because it will look for labels "NAME_B" in ds while it is set to "NAME_A":
        m = new_m()
        dsl = ds.copy()
        m.fit_predict(dsl, feature={'temperature': 'TEMP'}, inplace=True, name='NAME_A')
        pcmstats.quant(dsl, of='TEMP', using='NAME_B')

    with pytest.raises(TypeError):
        # This will raise an error if data are stored as dask array, because quantile cannot work on this type
        m = new_m()
        dsl = ds.copy() # pcmdata.load_argo() is using xr.open_mfdataset and return dask arrays
        m.fit_predict(dsl, feature={'temperature': 'TEMP'}, inplace=True)
        labelname = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset with LABELS
        pcmstats.quant(dsl, of='TEMP', using=labelname)

    # This must work:
    m = new_m()
    dsl = ds.copy().compute()
    m.fit_predict(dsl, feature={'temperature': 'TEMP'}, inplace=True)
    labelname = list(set(dsl.data_vars) - set(ds.data_vars))[0]  # Name of the new variable in the dataset with LABELS
    Q = pcmstats.quant(dsl, of='TEMP', using=labelname)
    # class: `xarray.DataArray` with shape(K, n_quantiles, N_z=n_features)
    assert isinstance(Q, xr.DataArray) == True, "Output must be a xarray.DataArray"


