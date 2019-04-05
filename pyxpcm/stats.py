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
        ds[name] = xr.concat(qlist, dim=dim_class)
    else:
        if found_class:
            qlist = xr.concat(qlist, dim=ds[dim_class])
        else:
            qlist = xr.concat(qlist, dim=dim_class)
        # qlist = xr.concat(qlist, dim=("N_CLASS"))
        # qlist = xr.concat(qlist, dim=("N_CLASS_%s)"%(qname)))
        return qlist