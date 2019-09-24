#!/bin/env python
# -*coding: UTF-8 -*-
#
# Provide some basic methods to compute statistic of classes
#
# Created by gmaze on 2017/12/05
__author__ = 'gmaze@ifremer.fr'

import os
import sys
import xarray as xr
import numpy as np
import dask.array
import warnings

from .pcmodel import xarray_accessor_pyXpcm

def quant(ds,
          of=None,
          using='LABEL',
          q=[0.05, 0.5, 0.95],
          inplace=False,
          dim=None,
          name='QUANT'):
    """Compute q-th quantiles of a dataArray for each PCM component

    Parameters
    ----------
    ds: :class:`xarray.DataSet`
        The dataset to work with
    of: str
        Name of the :class:`xarray.DataArray` to compute quantiles for.
    using: str
        Name of the :class:`xarray.DataArray` with classification labels to use.
    q: float in the range of [0,1] (or sequence of floats)
        Quantiles to compute, which must be between 0 and 1 inclusive.
    dim : str, optional
        Sampling Dimension over which to compute quantiles. Set to the first dimension by default.

    Returns
    -------
    Q: :class:`xarray.DataArray` with shape (K, n_quantiles, N_z=n_features)

    Examples
    --------
    ::
        from pyxpcm import stats as pcmstats
        ds = ds.compute()
        pcmstats.quant(ds, of='TEMP', using='PCM_LABELS')

    """
    if using not in ds.data_vars:
        raise ValueError(("Variable '%s' not found in this dataset") % (using))

    if of not in ds.data_vars:
        raise ValueError(("Variable '%s' not found in this dataset") % (of))

    # Fill in the dataset, otherwise the xarray.quantile doesn't work
    # ds = ds.compute()
    if isinstance(ds[of].data, dask.array.Array):
        raise TypeError("quant does not work for arrays stored as dask "
                        "arrays. Load the data via .compute() or .load() "
                        "prior to calling this method.")

    if not dim:
        # Assume the first dimension is the sampling dimension:
        dim = str(ds[of].dims[0])

    qlist = [] # list of quantiles to compute
    for label, group in ds.groupby(using):
        v = group[of].quantile(q, dim=dim)
        qlist.append(v)

    # Try to infer the dimension of the class components:
    # The dimension probably has the unique value of the labels:
    uniquelabels = np.unique(ds[using])
    found_class = False
    for thisdim in ds.dims:
        if len(ds[thisdim].values) == len(uniquelabels) and\
                np.array_equal(ds[thisdim].values, uniquelabels):
            dim_class = thisdim
            found_class = True
    if not found_class:
        dim_class = ("N_CLASS_%s")%(name)

    if inplace:
        da = xr.concat(qlist, dim=dim_class).rename(name)
        return ds.pyxpcm.add_inplace(da)
        # ds[name] = da

    else:
        if found_class:
            qlist = xr.concat(qlist, dim=ds[dim_class])
        else:
            qlist = xr.concat(qlist, dim=dim_class)
        return qlist

def robustness(ds, name='PCM_POST', classdimname='N_CLASS', inplace=False, outname='PCM_ROBUSTNESS'):
    """ Compute classification robustness

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            Input dataset

        name: str, default is 'PCM_POST'
            Name of the :class:`xarray.DataArray` with prediction probability (posteriors)

        classdimname: str, default is 'N_CLASS'
            Name of the dimension holding classes

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with robustness
            If True, return the input :class:`xarray.DataSet` with robustness added as a new :class:`xarray.DataArray`

        Returns
        -------
        :class:`xarray.DataArray`
            Robustness of the classification

        __author__: gmaze@ifremer.fr
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

    # Add labels to the dataset:
    if inplace:
        if outname in ds.data_vars:
            warnings.warn(("%s variable already in the dataset: overwriting") % (outname))
        return ds.pyxpcm.add_inplace(da)
        # ds[outname] = da
        # ds[outname].attrs['pyxpcm'] = "Automatically added by pyXpcm"
    else:
        return da

def robustness_digit(ds, name='PCM_POST', classdimname='N_CLASS', inplace=False, outname='PCM_ROBUSTNESS_CAT'):
    """ Digitize classification robustness

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            Input dataset

        name: str, default is 'PCM_POST'
            Name of the :class:`xarray.DataArray` with prediction probability (posteriors)

        classdimname: str, default is 'N_CLASS'
            Name of the dimension holding classes

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with robustness category
            If True, return the input :class:`xarray.DataSet` with robustness category added as a new :class:`xarray.DataArray`

        Returns
        -------
        :class:`xarray.DataArray`
            Robustness category of the classification

        __author__: gmaze@ifremer.fr
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
    if inplace:
        if outname in ds.data_vars:
            warnings.warn(("%s variable already in the dataset: overwriting") % (outname))
        return ds.pyxpcm.add_inplace(da)
        # ds[outname] = da
        # ds[outname].attrs['pyxpcm'] = "Automatically added by pyXpcm"

    else:
        return da