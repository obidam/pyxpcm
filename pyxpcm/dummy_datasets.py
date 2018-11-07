# -*coding: UTF-8 -*-
__author__ = 'gmaze@ifremer.fr'

from os.path import dirname, join
import xarray as xr

def load_argo(return_X_y=False):
    """Load and return a sample of Argo profiles on standard depth levels"""
    module_path = dirname(__file__)
    ncfile = 'argo_sample_test.nc'
    return xr.open_mfdataset(join(module_path, "data", ncfile))
