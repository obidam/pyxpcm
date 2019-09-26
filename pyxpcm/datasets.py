# -*coding: UTF-8 -*-
__author__ = 'gmaze@ifremer.fr'

from os.path import dirname, join
import numpy as np
import xarray as xr

def load_argo():
    """Load and return a sample of Argo profiles on standard depth levels"""
    module_path = dirname(__file__)
    ncfile = 'argo_sample.nc'
    ds = xr.open_mfdataset(join(module_path, "data", ncfile))
    ds.attrs = dict()
    ds.attrs['Prepared by'] = "G. Maze"
    ds.attrs['Institution'] = "Ifremer/LOPS"
    ds.attrs['Data DOI'] = "10.17882/42182"
    return ds

def load_isas15():
    """Load and return a sample of ISAS15 data"""
    module_path = dirname(__file__)
    ncfile = 'isas15_sample_test.nc'
    ds = xr.open_mfdataset(join(module_path, "data", ncfile), chunks={'latitude': 5, 'longitude': 5})
    ds['depth'] = -np.abs(ds['depth'])
    ds['SST'] = ds['TEMP'].isel(depth=0)
    # Data small enough to fit in memory on any computer
    ds = ds.chunk({'latitude': None, 'longitude': None})
    ds = ds.compute()
    return ds

def load_isas15series():
    """Load and return a sample of ISAS15 data timeseries"""
    module_path = dirname(__file__)
    ncfile = 'isas15series_sample_test.nc'
    ds = xr.open_dataset(join(module_path, "data", ncfile), chunks={'time':5, 'latitude': 5, 'longitude': 5})
    ds['depth'] = -np.abs(ds['depth'])
    ds['SST'] = ds['TEMP'].isel(depth=0)
    # Data small enough to fit in memory on any computer
    ds = ds.chunk({'latitude': None, 'longitude': None, 'time': None})
    ds = ds.compute()
    return ds
