# -*coding: UTF-8 -*-
__author__ = 'gmaze@ifremer.fr'

from os.path import dirname, join
import numpy as np
import xarray as xr

class argo():
    def __init__(self, what='sample'):
        self.data_root = join(dirname(__file__), 'data')
        categories = ['sample']
        if what not in categories:
            raise ValueError("I can't load a '%s' of Argo data" % what)
        else:
            self.category = what
        pass

    def load(self):
        """Load and return a sample of Argo profiles on standard depth levels"""
        if self.category == 'sample':
            ncfile = 'argo_sample_test.nc'
            ds = xr.open_mfdataset(join(self.data_root, ncfile))
            ds['DEPTH'].attrs['axis'] = 'Z'
            #todo I need to add these attributes directly into the netcdf file
            ds.attrs = dict()
            ds.attrs['Sample test prepared by'] = "G. Maze"
            ds.attrs['Institution'] = "Ifremer/LOPS"
            ds.attrs['Data source DOI'] = "10.17882/42182"
        return ds

class isas():
    def __init__(self, what='sample_snapshot', version='15'):
        self.data_root = join(dirname(__file__), 'data')
        self.version = version
        categories = ['sample_snapshot','sample_series']
        if what not in categories:
            raise ValueError("I can't load a '%s' of ISAS data" % what)
        else:
            self.category = what
        pass

    def load(self):
        """Load and return a sample of ISAS profiles on standard depth levels"""
        if self.category == 'sample_snapshot':
            ncfile = 'isas15_sample_test.nc'
            ds = xr.open_mfdataset(join(self.data_root, ncfile))
            ds['depth'] = -np.abs(ds['depth'])
            ds['depth'].attrs['axis'] = 'Z'
            ds['SST'] = ds['TEMP'].isel(depth=0)
            # Data small enough to fit in memory on any computer
            ds = ds.chunk({'latitude': None, 'longitude': None})
            ds = ds.compute()
        elif self.category == 'sample_series':
            ncfile = 'isas15series_sample_test.nc'
            ds = xr.open_mfdataset(join(self.data_root, ncfile))
            ds['depth'] = -np.abs(ds['depth'])
            ds['depth'].attrs['axis'] = 'Z'
            ds['SST'] = ds['TEMP'].isel(depth=0)
            # Data small enough to fit in memory on any computer
            ds = ds.chunk({'latitude': None, 'longitude': None, 'time': None})
            ds = ds.compute()
        return ds

def load_argo():
    """Load and return a sample of Argo profiles on standard depth levels"""
    raise ValueError('This function is deprecated, please use: pyxpcm.datasets.argo class')

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
