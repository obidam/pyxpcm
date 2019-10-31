#!/bin/env python
# -*coding: UTF-8 -*-
#
# Methods to be accessed through the xarray accessor or pcm "stat" space:
# m.stat.<method>
# ds.pyxpcm.<method>
#
# Created by gmaze on 2017/12/05
__author__ = 'gmaze@ifremer.fr'

import os
import sys
import xarray as xr
import numpy as np
import dask.array
import warnings
from .utils import docstring

def quantile(ds,
             q=0.5,
             of=None,
             using='PCM_LABELS',
             outname='PCM_QUANT',
             keep_attrs=False):
    """Compute q-th quantile of a :class:`xarray.DataArray` for each PCM components

        Parameters
        ----------
        q: float in the range of [0,1] (or sequence of floats)
            Quantiles to compute, which must be between 0 and 1 inclusive.

        of: str
            Name of the :class:`xarray.Dataset` variable to compute quantiles for.

        using: str
            Name of the :class:`xarray.Dataset` variable with classification labels to use.
            Use 'PCM_LABELS' by default.

        outname: 'PCM_QUANT' or str
            Name of the :class:`xarray.DataArray` with quantile

        keep_attrs: boolean, False by default
            Preserve ``of`` :class:`xarray.Dataset` attributes or not in the new quantile variable.

        Returns
        -------
        :class:`xarray.Dataset` with shape (K, n_quantiles, N_z=n_features)
        or
        :class:`xarray.DataArray` with shape (K, n_quantiles, N_z=n_features)

    """
    # Fill in the dataset, otherwise the xarray.quantile doesn't work
    if isinstance(ds[of].data, dask.array.Array):
        warnings.warn("quantile does not work for arrays stored as dask arrays. Loading array data via .load() ")
        ds[of].compute()
        ds[using].compute()

    if using not in ds.data_vars:
        raise ValueError(("Variable '%s' not found in this dataset") % (using))

    if of not in ds.data_vars:
        raise ValueError(("Variable '%s' not found in this dataset") % (of))

    # ID sampling dimensions for this array (all dimensions but those of LABELS)
    sampling_dims = ds[using].dims
    ds = ds.stack({'sampling': sampling_dims})
    qlist = []  # list of quantiles to compute
    for label, group in ds.groupby(using):
        da = group[of]
        if isinstance(da.data, dask.array.Array):
            da = da.compute()
        v = da.quantile(q, dim='sampling', keep_attrs=True)
        qlist.append(v)

    # Try to infer the dimension of the class components:
    # The dimension surely has the unique value in labels:
    l = ds[using].where(ds[using].notnull(), drop=True).values.flatten()
    uniquelabels = np.unique(l[~np.isnan(l)])
    found_class = False
    for thisdim in ds.dims:
        if len(ds[thisdim].values) == len(uniquelabels) and \
                np.array_equal(ds[thisdim].values, uniquelabels):
            dim_class = thisdim
            found_class = True
    if not found_class:
        dim_class = ("pcm_class_%s") % (outname)

    # Create xarray with all quantiles:
    if found_class:
        da = xr.concat(qlist, dim=ds[dim_class])
    else:
        da = xr.concat(qlist, dim=dim_class)
    da = da.rename(outname)
    da = da.unstack()

    # Output:
    if keep_attrs:
        da.attrs = ds[of].attrs
    return da

def robustness(ds, name='PCM_POST', classdimname='pcm_class', outname='PCM_ROBUSTNESS'):
    """ Compute classification robustness

        Parameters
        ----------
        name: str, default is 'PCM_POST'
            Name of the :class:`xarray.DataArray` with prediction probability (posteriors)

        classdimname: str, default is 'pcm_class'
            Name of the dimension holding classes

        outname: 'PCM_ROBUSTNESS' or str
            Name of the :class:`xarray.DataArray` with robustness

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with robustness
            If True, return the input :class:`xarray.Dataset` with robustness added as a new :class:`xarray.DataArray`

        Returns
        -------
        :class:`xarray.Dataset` if inplace=True
        or
        :class:`xarray.DataArray` if inplace=False

    """
    maxpost = ds[name].max(dim=classdimname)
    K = len(ds[classdimname])
    robust = (maxpost - 1. / K) * K / (K - 1.)

    id = dict()
    id[classdimname] = 0
    da = ds[name][id].rename(outname)
    da.values = robust
    da.attrs['long_name'] = 'PCM classification robustness'
    da.attrs['units'] = ''
    da.attrs['valid_min'] = 0
    da.attrs['valid_max'] = 1
    da.attrs['llh'] = ds[name].attrs['llh']

    #
    return da

def robustness_digit(ds, name='PCM_POST', classdimname='pcm_class', outname='PCM_ROBUSTNESS_CAT'):
    """ Digitize classification robustness

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            Input dataset

        name: str, default is 'PCM_POST'
            Name of the :class:`xarray.DataArray` with prediction probability (posteriors)

        classdimname: str, default is 'pcm_class'
            Name of the dimension holding classes

        outname: 'PCM_ROBUSTNESS_CAT' or str
            Name of the :class:`xarray.DataArray` with robustness categories

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with robustness
            If True, return the input :class:`xarray.Dataset` with robustness categories
            added as a new :class:`xarray.DataArray`

        Returns
        -------
        :class:`xarray.Dataset` if inplace=True
        or
        :class:`xarray.DataArray` if inplace=False
    """
    maxpost = ds[name].max(dim=classdimname)
    K = len(ds[classdimname])
    robust = (maxpost - 1. / K) * K / (K - 1.)
    Plist = [0, 0.33, 0.66, 0.9, .99, 1]
    rowl0 = ('Unlikely', 'As likely as not', 'Likely', 'Very Likely', 'Virtually certain')
    robust_id = np.digitize(robust, Plist) - 1

    id = dict()
    id[classdimname] = 0
    da = ds[name][id].rename(outname)
    da.values = robust_id
    da.attrs['long_name'] = 'PCM classification robustness category'
    da.attrs['units'] = ''
    da.attrs['valid_min'] = 0
    da.attrs['valid_max'] = 4
    da.attrs['llh'] = ds[name].attrs['llh']
    da.attrs['bins'] = Plist
    da.attrs['legend'] = rowl0

    # Add labels to the dataset:
    return da

class _StatMethods(object):
    """
        Enables use of pyxpcm.stat functions as attributes on a PCM object.
    """

    def __init__(self, m):
        self._pcm = m

    def __call__(self, **kwargs):
        raise ValueError("pyxpcm.stat cannot be called directly. Use one of the statistics methods: quantile, robustness, robustness_digit")

    @docstring(quantile.__doc__)
    def quantile(self, *ags, **kwargs):
        return quantile(*ags, **kwargs)

    @docstring(robustness.__doc__)
    def robustness(self, *ags, **kwargs):
        return robustness(*ags, **kwargs)

    @docstring(robustness_digit.__doc__)
    def robustness_digit(self, *ags, **kwargs):
        return robustness_digit(*ags, **kwargs)
