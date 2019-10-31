# -*coding: UTF-8 -*-
"""

    Useful for documentation and to play with pyXpcm


    Data files should be hosted on another repo

"""

# import os
from os.path import dirname, join
import numpy as np
import xarray as xr

#todo Re-factor tutorial dataset loading process following ideas from Xarray/Seaborn
# https://github.com/pydata/xarray/blob/master/xarray/tutorial.py

#todo Migrate data files to another repo

def open_dataset(name):
    """ Open a toy dataset from the pyXpcm distrib

        Parameters
        ----------
        name : str
            Name of the dataset to load among:

                - `dummy` (depth,sample) dummy array
                - `argo`  (depth,sample) real Argo data sample
                - `isas_snapshot` (depth,latitude,longitude) real gridded product
                - `isas_series` (depth,latitude,longitude,time) real gridded product time series

        Returns
        -------
        :class:`xarray.Dataset`

    """

    if name == 'argo':
        acc = argo(what='sample')

    elif name == 'isas_snapshot':
        acc = isas(what='sample_snapshot')

    elif name == 'isas_series':
        acc = isas(what='sample_series')

    elif name == 'dummy':
        acc = dummy()

    else:
        raise ValueError("Don't know this tutorial dataset")

    return acc

class dummy():

    def load(self, Np=1000, Nz=50):
        z = np.linspace(0, -500, Nz)
        ds = xr.Dataset({
            'TEMP': xr.DataArray(np.random.rand(Np, Nz),
                                 dims=['n_prof', 'depth'], coords={'depth': z}),
            'PSAL': xr.DataArray(np.random.rand(Np, Nz),
                                 dims=['n_prof', 'depth'], coords={'depth': z})
            })
        ds['depth'].attrs['axis'] = 'Z'
        ds['depth'].attrs['units'] = 'meters'
        ds['depth'].attrs['positive'] = 'up'
        ds['TEMP'].attrs['feature_name'] = 'temperature'
        ds['PSAL'].attrs['feature_name'] = 'salinity'
        ds.attrs['comment'] = "Dummy fields with random values"
        return ds

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
            ds['DEPTH'].attrs['units'] = 'meters'
            ds['DEPTH'].attrs['positive'] = 'up'
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
            ds['depth'].attrs['units'] = 'meters'
            ds['depth'].attrs['positive'] = 'up'
            ds['SST'] = ds['TEMP'].isel(depth=0)
            # Data small enough to fit in memory on any computer
            ds = ds.chunk({'latitude': None, 'longitude': None})
            # ds = ds.compute()

        elif self.category == 'sample_series':
            ncfile = 'isas15series_sample_test.nc'
            ds = xr.open_mfdataset(join(self.data_root, ncfile))
            ds['depth'] = -np.abs(ds['depth'])
            ds['depth'].attrs['axis'] = 'Z'
            ds['depth'].attrs['units'] = 'meters'
            ds['depth'].attrs['positive'] = 'up'
            ds['SST'] = ds['TEMP'].isel(depth=0)
            # Data small enough to fit in memory on any computer
            ds = ds.chunk({'latitude': None, 'longitude': None, 'time': None})
            # ds = ds.compute()
        return ds

def load_argo():
    """Load and return a sample of Argo profiles on standard depth levels"""
    raise ValueError('This function is deprecated, please use: pyxpcm.tutorial.open_dataset')

def load_isas15():
    """Load and return a sample of ISAS15 data"""
    raise ValueError('This function is deprecated, please use: pyxpcm.tutorial.open_dataset')

def load_isas15series():
    """Load and return a sample of ISAS15 data timeseries"""
    raise ValueError('This function is deprecated, please use: pyxpcm.tutorial.open_dataset')

