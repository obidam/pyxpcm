#!/bin/env python
# -*coding: UTF-8 -*-
"""
Provide accessor to enhance interoperability between :mod:`xarray` and :mod:`pyxpcm`.

Provide a scope named ``pyxpcm`` as accessor to :class:`xarray.Dataset` objects.

"""
# Created by gmaze on 2019-10-02

# Note that because of https://github.com/pydata/xarray/issues/3268, nothing happens in place here. So it should always goes like (even with `inplace=True`):
#
#     ds = ds.pyxpcm.<method>()

import os
import sys
import warnings
import numpy as np
import xarray as xr
import dask
from .models import pcm, PCMFeatureError
from . import stat
# from .utils import docstring

# Decorators
def pcm_method(func):
    #todo Decorator that directly map PCM functions on xarray accessor
    func.__doc__ = getattr(pcm, func.__name__).__doc__
    return func

def pcm_stat_method(func):
    #todo Decorator that directly map PCM functions on xarray accessor
    func.__doc__ = getattr(stat, func.__name__).__doc__
    return func

@xr.register_dataset_accessor('pyxpcm')
class pyXpcmDataSetAccessor:
    """

        Class registered under scope ``pyxpcm`` to access :class:`xarray.Dataset` objects.

     """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._added = list() # Will record all new variables added by pyXpcm
        self._dims = list(xarray_obj.dims.keys()) # Store the initial list of dimensions

    def __id_feature_name(self, this_pcm, feature):
        """Identify the dataset variable name to be used for a given feature name

            feature must be a dictionary or None for automatic discovery
        """
        feature_name_found = False

        for feature_in_pcm in feature:
            if feature_in_pcm not in this_pcm._props['features']:
                msg = ("Feature '%s' not set in this PCM")%(feature_in_pcm)
                raise PCMFeatureError(msg)

            feature_in_ds = feature[feature_in_pcm]

            if feature_in_ds:
                feature_name_found = feature_in_ds in self._obj.data_vars

            if not feature_name_found:
                feature_name_found = feature_in_pcm in self._obj.data_vars
                feature_in_ds = feature_in_pcm

            if not feature_name_found:
                # Look for the feature in the dataset data variables attributes
                for v in self._obj.data_vars:
                    if ('feature_name' in self._obj[v].attrs) and (self._obj[v].attrs['feature_name'] is feature_in_pcm):
                        feature_in_ds = v
                        feature_name_found = True
                        continue

            if not feature_name_found:
                msg = ("Feature '%s' not found in this dataset. You may want to add the 'feature_name' "
                                  "attribute to the variable you'd like to use or provide a dictionnary")%(feature_in_pcm)
                raise PCMFeatureError(msg)
            elif this_pcm._debug:
                print(("\tIdying '%s' as '%s' in this dataset") % (feature_in_pcm, feature_in_ds))

        return feature_in_ds

    def add(self, da):
        """Add a :class:`xarray.DataArray` to this :class:`xarray.Dataset`"""
        if da.name in self._obj.data_vars:
            warnings.warn(("%s variable already in the dataset: overwriting") % (da.name))

        # Add pyXpcm tracking clue to this DataArray:
        da.attrs['_pyXpcm_cleanable'] = True

        # Add it to the DataSet:
        self._obj[da.name] = da

        # Update internal list of added variables:
        self._added.append(da.name)
        return self._obj

    def drop_all(self):
        """ Remove :class:`xarray.DataArray` created with pyXpcm front this :class:`xarray.Dataset`"""
        def drop_dims(this_ds, dims):
            return this_ds.drop_dims([k for k in this_ds.dims if k not in dims])
        ds = drop_dims(self._obj, self._dims)
        return ds.drop([
            k for k, v in ds.variables.items()
            if v.attrs.get("_pyXpcm_cleanable")
        ])

    def split(self):
        """Split pyXpcm variables from the original :class:`xarray.Dataset`

            Returns
            -------
            :class:`xarray.Dataset`, :class:`xarray.Dataset`
                Two DataSest: one with pyXpcm variables, one with the original DataSet
        """
        _ds = self._obj[[
            k for k, v in self._obj.variables.items()
            if v.attrs.get("_pyXpcm_cleanable")
        ]].copy(deep=True)
        return _ds, self.drop_all()

    def feature_dict(self, this_pcm, features=None):
        """ Return dictionary of features for this :class:`xarray.Dataset` and a PCM

            Parameters
            ----------
            pcm : :class:`pyxpcm.pcmmodel.pcm`

            features : dict
                Keys are PCM feature name, Values are corresponding :class:`xarray.Dataset` variable names

            Returns
            -------
            dict()
                Dictionary where keys are PCM feature names and values the corresponding :class:`xarray.Dataset` variables

        """
        features_dict = dict()
        for feature_in_pcm in this_pcm._props['features']:
            if features == None:
                feature_in_ds = self.__id_feature_name(this_pcm, {feature_in_pcm: None})
            elif feature_in_pcm not in features:
                raise PCMFeatureError("%s feature not defined" % feature_in_pcm)
            else:
                feature_in_ds = features[feature_in_pcm]
                if feature_in_ds not in self._obj.data_vars:
                    raise PCMFeatureError("Feature %s not in this dataset as %s" % (feature_in_pcm, feature_in_ds))
            features_dict[feature_in_pcm] = feature_in_ds

        # if features:
        #     features_dict = dict()
        #     for feature_in_pcm in features:
        #         feature_in_ds = features[feature_in_pcm]
        #         if not feature_in_ds:
        #             feature_in_ds = self.__id_feature_name(pcm, {feature_in_pcm: None})
        #         features_dict[feature_in_pcm] = feature_in_ds
        # else:
        #     features_dict = dict()
        #     for feature_in_pcm in pcm._props['features']:
        #         feature_in_ds = self.__id_feature_name(pcm, {feature_in_pcm: None})
        #         features_dict[feature_in_pcm] = feature_in_ds

        # Re-order the dictionary to match the PCM set order:
        for key in this_pcm._props['features']:
            features_dict[key] = features_dict.pop(key)

        return features_dict

    def sampling_dim(self, this_pcm, features=None, dim=None):
        """ Return the list of dimensions to be stacked for sampling

            Parameters
            ----------
            pcm : :class:`pyxpcm.pcm`

            features : None (default) or dict()
                Keys are PCM feature name, Values are corresponding :class:`xarray.Dataset` variable names.
                It set to None, all PCM features are used.

            dim : None (default) or str()
                The :class:`xarray.Dataset` dimension to use as vertical axis in all features.
                If set to None, it is automatically set to the dimension with an attribute ``axis`` set to ``Z``.

            Returns
            -------
            dict()
                Dictionary where keys are :class:`xarray.Dataset` variable names of features and values are another
                dictionary with the list of sampling dimension in DIM_SAMPLING key and the name of the vertical axis in
                the DIM_VERTICAL key.

        """

        feature_dict = self.feature_dict(this_pcm, features=features)
        SD = dict()

        for feature_name_in_pcm in feature_dict:
            feature_name_in_ds = feature_dict[feature_name_in_pcm]
            da = self._obj[feature_name_in_ds]
            SD[feature_name_in_ds] = dict()

            # Is this a thick array or a slice ?
            is_slice = np.all(this_pcm._props['features'][feature_name_in_pcm] is None)

            if is_slice:
                # No vertical dimension to use, simple stacking
                sampling_dims = list(da.dims)
                SD[feature_name_in_ds]['DIM_SAMPLING'] = sampling_dims
                SD[feature_name_in_ds]['DIM_VERTICAL'] = None
            else:
                if not dim:
                    # Try to infer the vertical dimension name looking for the CF 'axis' attribute in all dimensions of the array
                    dim_found = False
                    for this_dim in da.dims:
                        if ('axis' in da[this_dim].attrs) and (da[this_dim].attrs['axis'] == 'Z'):
                            dim = this_dim
                            dim_found = True
                    if not dim_found:
                        raise PCMFeatureError("You must specify a vertical dimension name: "\
                                              "use argument 'dim' or "\
                                              "specify DataSet dimension the attribute 'axis' to 'Z' (CF1.6)")
                elif dim not in da.dims:
                    raise ValueError("Vertical dimension %s not found in this DataArray" % dim)

                sampling_dims = list(da.dims)
                sampling_dims.remove(dim)
                SD[feature_name_in_ds]['DIM_SAMPLING'] = sampling_dims
                SD[feature_name_in_ds]['DIM_VERTICAL'] = dim

        return SD

    def mask(self, this_pcm, features=None, dim=None):
        """ Create a mask where all PCM features are defined

            Create a mask where all feature profiles are not null
            over the PCM feature axis.

            Parameters
            ----------
            :class:`pyxpcm.pcmmodel.pcm`

            features : dict()
                Definitions of this_pcm features in the :class:`xarray.Dataset`.
                If not specified or set to None, features are identified
                using :class:`xarray.DataArray` attributes 'feature_name'.

            dim : str
                Name of the vertical dimension in the :class:`xarray.Dataset`.
                If not specified or set to None, dim is identified as the
                :class:`xarray.DataArray` variables with attributes 'axis' set to 'z'.

            Returns
            -------
            :class:`xarray.DataArray`

        """
        feature_dict = self.feature_dict(this_pcm, features=features)
        SD = self.sampling_dim(this_pcm, dim=dim, features=features)
        M = list()
        for feature_name_in_this_pcm in feature_dict:
            feature_name_in_ds = feature_dict[feature_name_in_this_pcm]
            da = self._obj[feature_name_in_ds]

            # Is this a thick array or a slice ?
            is_slice = np.all(this_pcm._props['features'][feature_name_in_this_pcm] == None)

            if not is_slice:
                dim = SD[feature_name_in_ds]['DIM_VERTICAL']
                z_top = np.max(this_pcm._props['features'][feature_name_in_this_pcm])
                z_bto = np.min(this_pcm._props['features'][feature_name_in_this_pcm])

                # Nz = len((self._obj[dim].where(self._obj[dim] >= z_bto, drop=True)\
                #                         .where(self._obj[dim] <= z_top, drop=True)).notnull())
                z = self._obj[dim]
                z_ok = np.ma.masked_inside(z, z_bto, z_top, copy=True).mask
                Nz = np.count_nonzero(z_ok == True)
                mask = self._obj[feature_name_in_ds][{dim:z_ok}].notnull().sum(dim=dim) == Nz

                # mask = self._obj[feature_name_in_ds]\
                #             .where(z >= z_bto)\
                #             .where(z <= z_top).notnull().sum(dim=dim) == Nz

                # mask = self._obj[feature_name_in_ds].where(
                #     self._obj[dim]>=z_bto).notnull().sum(dim=dim) == len(np.where(self._obj[dim]>=z_bto)[0])
            else:
                mask = self._obj[feature_name_in_ds].notnull()
            mask = mask.rename('pcm_MASK')
            M.append(mask)
        mask = xr.concat(M, dim='n_features')
        mask = mask.sum(dim='n_features')
        mask = mask == this_pcm.F
        return mask

    @pcm_method
    def fit(self, this_pcm, **kwargs):
        this_pcm.fit(self._obj, **kwargs)

    @pcm_method
    def predict(self, this_pcm, inplace=False, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.predict` """
        da = this_pcm.predict(self._obj, **kwargs)
        if inplace:
            return self.add(da)
        else:
            return da

    @pcm_method
    def fit_predict(self, this_pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.fit_predict` """
        return this_pcm.fit_predict(self._obj, **kwargs)

    @pcm_method
    def predict_proba(self, this_pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.predict_proba` """
        return this_pcm.predict_proba(self._obj, **kwargs)

    @pcm_method
    def score(self, this_pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.score` """
        return this_pcm.score(self._obj, **kwargs)

    @pcm_method
    def bic(self, this_pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.bic` """
        return this_pcm.bic(self._obj, **kwargs)

    @pcm_stat_method
    def quantile(self, this_pcm, inplace=False, **kwargs):
        """ Map this :class:`xarray.Dataset` on :meth:`pyxpcm.pcm.stat.quantile` """
        da = this_pcm.stat.quantile(self._obj, **kwargs)
        if inplace:
            return self.add(da)
        else:
            return da

    @pcm_stat_method
    def robustness(self, this_pcm, inplace=False, **kwargs):
        """ Map this :class:`xarray.Dataset` on :meth:`pyxpcm.pcm.stat.robustness` """
        da = this_pcm.stat.robustness(self._obj, **kwargs)
        if inplace:
            return self.add(da)
        else:
            return da

    @pcm_stat_method
    def robustness_digit(self, this_pcm, inplace=False, **kwargs):
        """ Map this :class:`xarray.Dataset` on :meth:`pyxpcm.pcm.stat.robustness_digit` """
        da = this_pcm.stat.robustness_digit(self._obj, **kwargs)
        if inplace:
            return self.add(da)
        else:
            return da
