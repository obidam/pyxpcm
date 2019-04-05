# -*coding: UTF-8 -*-
__author__ = 'gmaze@ifremer.fr'

from os.path import dirname, join
import xarray as xr

def load_argo():
    """Load and return a sample of Argo profiles on standard depth levels"""
    module_path = dirname(__file__)
    ncfile = 'argo_sample_test.nc'
    return xr.open_mfdataset(join(module_path, "data", ncfile))

def load_isas15():
    """Load and return a sample of ISAS15 data"""
    # This was generated as:

    # ds = xr.open_dataset(
    #     '/home/datawork-lops-oh/ISAS/ISAS_USERS/ANA_ISAS15_DM/field/2005/ISAS15_DM_20051115_fld_TEMP.nc')
    # ds = ds.where(ds.longitude >= -70, drop=True)\
    #     .where(ds.longitude <= -40, drop=True)\
    #     .where(ds.latitude >= 30, drop=True)\
    #     .where(ds.latitude <= 50, drop=True)
    # ds = ds.isel(time=0)
    # ds = ds.drop('time')
    # ds.to_netcdf('isas15_sample_test.nc')
    module_path = dirname(__file__)
    ncfile = 'isas15_sample_test.nc'
    return xr.open_mfdataset(join(module_path, "data", ncfile), chunks={'latitude': 5, 'longitude':5})
