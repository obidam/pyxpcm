# -*coding: UTF-8 -*-
"""

    Useful for documentation and to play with pyXpcm

    Data files should be hosted on another repo

"""

import os
# from os.path import dirname, join
import numpy as np
import xarray as xr

import hashlib
from urllib.request import urlretrieve

_default_cache_dir = os.sep.join(("~", ".pyxpcm_tutorial_data"))

def open_dataset(name):
    """ Open a dataset from the pyXpcm online data repository (requires internet).

        If a local copy is found then always use that to avoid network traffic.

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
        acc = argo_loader(what='sample')

    elif name == 'isas_snapshot':
        acc = isas_loader(what='sample_snapshot')

    elif name == 'isas_series':
        acc = isas_loader(what='sample_series')

    elif name == 'dummy':
        acc = dummy()

    elif name == 'orsi':
        acc = orsi()

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

class argo_loader():
    def __init__(self, what='sample'):
        categories = ['sample']
        if what not in categories:
            raise ValueError("I can't load a '%s' of Argo data" % what)
        else:
            self.category = what
        pass

    def load(self):
        """Load and return a sample of Argo profiles on standard depth levels"""
        if self.category == 'sample':
            ncfile = 'argo_sample'
            ds = _open_dataset(ncfile)
            #todo I need to add these attributes directly into the netcdf file
            ds['DEPTH'].attrs['axis'] = 'Z'
            ds['DEPTH'].attrs['units'] = 'meters'
            ds['DEPTH'].attrs['positive'] = 'up'
            ds.attrs = dict()
            ds.attrs['Sample test prepared by'] = "G. Maze"
            ds.attrs['Institution'] = "Ifremer/LOPS"
            ds.attrs['Data source DOI'] = "10.17882/42182"
        return ds

class isas_loader():
    def __init__(self, what='sample_snapshot', version='15'):
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
            ncfile = 'isas15_sample'
            ds = _open_dataset(ncfile)
            #todo I need to add these attributes directly into the netcdf file
            ds['depth'] = -np.abs(ds['depth'])
            ds['depth'].attrs['axis'] = 'Z'
            ds['depth'].attrs['units'] = 'meters'
            ds['depth'].attrs['positive'] = 'up'
            ds['SST'] = ds['TEMP'].isel(depth=0)
            ds = ds.chunk({'latitude': None, 'longitude': None})

        elif self.category == 'sample_series':
            ncfile = 'isas15series_sample'
            ds = _open_dataset(ncfile)
            #todo I need to add these attributes directly into the netcdf file
            ds['depth'] = -np.abs(ds['depth'])
            ds['depth'].attrs['axis'] = 'Z'
            ds['depth'].attrs['units'] = 'meters'
            ds['depth'].attrs['positive'] = 'up'
            ds['SST'] = ds['TEMP'].isel(depth=0)
            ds = ds.chunk({'latitude': None, 'longitude': None, 'time': None})

        return ds

class orsi():
    def load(self):
        """Load path of ORSI fronts"""
        ncfile = 'ORSIfronts'
        ds = _open_dataset(ncfile)
        return ds

#######
# This is heavily borrowed/copied from https://github.com/pydata/xarray/blob/master/xarray/tutorial.py

def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()

# idea borrowed from Seaborn
def _open_dataset(
                name,
                cache=True,
                cache_dir=_default_cache_dir,
                github_url="https://github.com/obidam/pyxpcm-data",
                branch="master",
                **kws,
                ):
    """
    Open a dataset from the pyXpcm online data repository (requires internet).
    If a local copy is found then always use that to avoid network traffic.

        Parameters
        ----------
        name : str
           Name of the netcdf file containing the dataset
           ie. 'argo_sample'
        cache_dir : string, optional
            The directory in which to search for and write cached data.
        cache : boolean, optional
            If True, then cache data locally for use on subsequent calls
        github_url : string
            Github repository where the data is stored
        branch : string
            The git branch to download from
        kws : dict, optional
            Passed to xarray.open_dataset

        Returns
        -------
        :class:`xarray.Dataset`

    """
    longdir = os.path.expanduser(cache_dir)
    fullname = name + ".nc"
    localfile = os.sep.join((longdir, fullname))
    md5name = name + ".md5"
    md5file = os.sep.join((longdir, md5name))

    if not os.path.exists(localfile):

        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not os.path.isdir(longdir):
            os.mkdir(longdir)

        url = "/".join((github_url, "raw", branch, fullname))
        urlretrieve(url, localfile)
        url = "/".join((github_url, "raw", branch, md5name))
        urlretrieve(url, md5file)

        localmd5 = file_md5_checksum(localfile)
        with open(md5file, "r") as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            os.remove(localfile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise OSError(msg)

    ds = xr.open_dataset(localfile, **kws)

    if not cache:
        ds = ds.load()
        os.remove(localfile)

    return ds