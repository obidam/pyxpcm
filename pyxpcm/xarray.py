#!/bin/env python
# -*coding: UTF-8 -*-
"""
Provide accessor to enhance interoperability between :mod:`xarray` and :mod:`pyxpcm`.

Provide a scope named ``pyxpcm`` as accessor to :class:`xarray.Dataset` objects.

Note that because of https://github.com/pydata/xarray/issues/3268, nothing happens in place here. So it should always goes like (even with `inplace=True`):

    ds = ds.pyxpcm.<method>()

"""
# Created by gmaze on 2019-10-02


import os
import sys
import numpy as np
import xarray as xr
import dask
from .pcmodel import pcm, PCMFeatureError
import warnings

# Decorators
def pcm_method(func):
    #todo Decorator that directly map PCM functions on xarray accessor
    #todo Follow doctring from  PCM functions to xarray accessor
    return func

@xr.register_dataset_accessor('pyxpcm')
class pyXpcmDataSetAccessor:
    """

        Class registered under scope ``pyxpcm`` to access :class:`xarray.Dataset` objects.

     """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __id_feature_name(self, pcm, feature):
        """Identify the dataset variable name to be used for a given feature name

            feature must be a dictionary or None for automatic discovery
        """
        feature_name_found = False

        for feature_in_pcm in feature:
            if feature_in_pcm not in pcm._props['features']:
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
            elif pcm._debug:
                print(("\tIdying '%s' as '%s' in this dataset") % (feature_in_pcm, feature_in_ds))

        return feature_in_ds

    def add(self, da):
        """ Add a new :class:`xarray.DataArray` to this :class:`xarray.Dataset` """

        # Add pyXpcm tracking clues:
        da.attrs['comment'] = "Automatically added by pyXpcm"

        #
        # vname = da.name
        self._obj[da.name] = da
        return self._obj

    def clean(self):
        """ Remove all variables created with pyXpcm front this :class:`xarray.Dataset` """
        # See add() method to identify these variables.
        for vname in self._obj.data_vars:
            if ("comment" in self._obj[vname].attrs) \
                and (self._obj[vname].attrs['comment'] == "Automatically added by pyXpcm"):
                self._obj = self._obj.drop(vname)
        return self._obj

    def feature_dict(self, pcm, features=None):
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
        for feature_in_pcm in pcm._props['features']:
            if features == None:
                feature_in_ds = self.__id_feature_name(pcm, {feature_in_pcm: None})
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
        for key in pcm._props['features']:
            features_dict[key] = features_dict.pop(key)

        return features_dict

    def sampling_dim(self, pcm, features=None, dim=None):
        """ Return the list of dimensions to be stacked for sampling

            Parameters
            ----------
            pcm : :class:`pyxpcm.pcm`

            features : None (default) or dict()
                Keys are PCM feature name, Values are corresponding :class:`xarray.Dataset` variable names.
                It set to None, all PCM features are used.

            dim : None (default) or str()
                The :class:`xarray.Dataset` dimension to use as vertical axis in all features.
                If set to None, it is automatically set to the dimension with an atribute ``axis`` set to ``Z``.

            Returns
            -------
            dict()
                Dictionary where keys are :class:`xarray.Dataset` variable names of features and values are another
                dictionary with the list of sampling dimension in DIM_SAMPLING key and the name of the vertical axis in
                the DIM_VERTICAL key.


        """

        feature_dict = self.feature_dict(pcm, features=features)
        SD = dict()

        for feature_name_in_pcm in feature_dict:
            feature_name_in_ds = feature_dict[feature_name_in_pcm]
            da = self._obj[feature_name_in_ds]
            SD[feature_name_in_ds] = dict()

            # Is this a thick array or a slice ?
            is_slice = np.all(pcm._props['features'][feature_name_in_pcm] == None)

            if is_slice:
                # No vertical dimension to use, simple stacking
                sampling_dims = list(da.dims)
                SD[feature_name_in_ds]['DIM_SAMPLING'] = sampling_dims
                SD[feature_name_in_ds]['DIM_VERTICAL'] = None
            else:
                if not dim:
                    # Try to infer the vertical dimension name looking for the CF 'axis' attribute in all dimensions
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

    def mask(self, pcm, features=None, dim=None):
        """ Create a mask where all PCM features are defined

            Create a mask where all feature profiles are not null
            over the PCM feature axis.

            Parameters
            ----------
            :class:`pyxpcm.pcmmodel.pcm`

            features: dict()
                Definitions of PCM features in the :class:`xarray.Dataset`.
                If not specified or set to None, features are identified
                using :class:`xarray.DataArray` attributes 'feature_name'.

            dim: str
                Name of the vertical dimension in the :class:`xarray.Dataset`.
                If not specified or set to None, dim is identified as the
                :class:`xarray.DataArray` variables with attributes 'axis' set to 'z'.

            Returns
            -------
            :class:`xarray.DataArray`

        """
        feature_dict = self.feature_dict(pcm, features=features)
        SD = self.sampling_dim(pcm, dim=dim, features=features)
        M = list()
        for feature_name_in_pcm in feature_dict:
            feature_name_in_ds = feature_dict[feature_name_in_pcm]
            da = self._obj[feature_name_in_ds]

            # Is this a thick array or a slice ?
            is_slice = np.all(pcm._props['features'][feature_name_in_pcm] == None)

            if not is_slice:
                dim = SD[feature_name_in_ds]['DIM_VERTICAL']
                z_top = np.max(pcm._props['features'][feature_name_in_pcm])
                z_bto = np.min(pcm._props['features'][feature_name_in_pcm])

                Nz = len((self._obj[dim].where(self._obj[dim] >= z_bto, drop=True)\
                                        .where(self._obj[dim] <= z_top, drop=True)).notnull())
                mask = self._obj[feature_name_in_ds]\
                            .where(self._obj[dim] >= z_bto)\
                            .where(self._obj[dim] <= z_top).notnull().sum(dim=dim) == Nz

                # mask = self._obj[feature_name_in_ds].where(
                #     self._obj[dim]>=z_bto).notnull().sum(dim=dim) == len(np.where(self._obj[dim]>=z_bto)[0])
            else:
                mask = self._obj[feature_name_in_ds].notnull()
            mask = mask.rename('PCM_MASK')
            M.append(mask)
        mask = xr.concat(M, dim='n_features')
        mask = mask.sum(dim='n_features')
        mask = mask == pcm.F
        return mask

    def quantile(self,
                  q,
                  of=None,
                  using='PCM_LABELS',
                  outname='PCM_QUANT',
                  inplace=True,
                  keep_attrs=False):
        """Compute q-th quantile of a :class:`xarray.DataArray` for each PCM components

            Parameters
            ----------
            float in the range of [0,1] (or sequence of floats)
                Quantiles to compute, which must be between 0 and 1 inclusive.

            of: str
                Name of the :class:`xarray.Dataset` variable to compute quantiles for.

            using: str
                Name of the :class:`xarray.Dataset` variable with classification labels to use.
                Use 'PCM_LABELS' by default.

            outname: 'PCM_QUANT' or str
                Name of the :class:`xarray.DataArray` with quantile

            inplace: boolean, True by default
                If True, return the input :class:`xarray.Dataset` with quantile variable added as a new :class:`xarray.DataArray`
                If False, return a :class:`xarray.DataArray` with quantile

            keep_attrs: boolean, False by default
                Preserve ``of`` :class:`xarray.Dataset` attributes or not in the new quantile variable.

            Returns
            -------
            :class:`xarray.Dataset` with shape (K, n_quantiles, N_z=n_features)
            or
            :class:`xarray.DataArray` with shape (K, n_quantiles, N_z=n_features)

        """

        if using not in self._obj.data_vars:
            raise ValueError(("Variable '%s' not found in this dataset") % (using))

        if of not in self._obj.data_vars:
            raise ValueError(("Variable '%s' not found in this dataset") % (of))

        # Fill in the dataset, otherwise the xarray.quantile doesn't work
        # ds = ds.compute()
        if isinstance(self._obj[of].data, dask.array.Array):
            raise TypeError("quant does not work for arrays stored as dask "
                            "arrays. Load the data via .compute() or .load() "
                            "prior to calling this method.")

        # ID sampling dimensions for this array (all dimensions but those of LABELS)
        sampling_dims = self._obj[using].dims
        ds = self._obj.stack({'sampling': sampling_dims})
        qlist = []  # list of quantiles to compute
        for label, group in ds.groupby(using):
            v = group[of].quantile(q, dim='sampling', keep_attrs=True)
            qlist.append(v)

        # Try to infer the dimension of the class components:
        # The dimension surely has the unique value in labels:
        l = self._obj[using].where(self._obj[using].notnull(), drop=True).values.flatten()
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
        if inplace:
            return self.add(da)
        else:
            return da

    def robustness(self, name='PCM_POST', classdimname='pcm_class', outname='PCM_ROBUSTNESS', inplace=True):
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
        maxpost = self._obj[name].max(dim=classdimname)
        K = len(self._obj[classdimname])
        robust = (maxpost - 1. / K) * K / (K - 1.)

        id = dict()
        id[classdimname] = 0
        da = self._obj[name][id].rename(outname)
        da.values = robust
        da.attrs['long_name'] = 'PCM classification robustness'
        da.attrs['units'] = ''
        da.attrs['valid_min'] = 0
        da.attrs['valid_max'] = 1
        da.attrs['llh'] = self._obj[name].attrs['llh']

        # Add labels to the dataset:
        if inplace:
            if outname in self._obj.data_vars:
                warnings.warn(("%s variable already in the dataset: overwriting") % (outname))
            return self.add(da)
        else:
            return da

    def robustness_digit(self, name='PCM_POST', classdimname='pcm_class', outname='PCM_ROBUSTNESS_CAT', inplace=True):
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
        maxpost = self._obj[name].max(dim=classdimname)
        K = len(self._obj[classdimname])
        robust = (maxpost - 1. / K) * K / (K - 1.)
        Plist = [0, 0.33, 0.66, 0.9, .99, 1]
        rowl0 = ('Unlikely', 'As likely as not', 'Likely', 'Very Likely', 'Virtually certain')
        robust_id = np.digitize(robust, Plist) - 1

        id = dict()
        id[classdimname] = 0
        da = self._obj[name][id].rename(outname)
        da.values = robust_id
        da.attrs['long_name'] = 'PCM classification robustness category'
        da.attrs['units'] = ''
        da.attrs['valid_min'] = 0
        da.attrs['valid_max'] = 4
        da.attrs['llh'] = self._obj[name].attrs['llh']
        da.attrs['bins'] = Plist
        da.attrs['legend'] = rowl0

        # Add labels to the dataset:
        if inplace:
            if outname in self._obj.data_vars:
                warnings.warn(("%s variable already in the dataset: overwriting") % (outname))
            return self.add(da)
        else:
            return da

    @pcm_method
    def fit(self, pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :meth:`pyxpcm.pcm.fit` """
        return pcm.fit(self._obj, **kwargs)

    @pcm_method
    def predict(self, pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.predict` """
        return pcm.predict(self._obj, **kwargs)

    @pcm_method
    def fit_predict(self, pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.fit_predict` """
        return pcm.fit_predict(self._obj, **kwargs)

    @pcm_method
    def predict_proba(self, pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.predict_proba` """
        return pcm.predict_proba(self._obj, **kwargs)

    @pcm_method
    def score(self, pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.score` """
        return pcm.score(self._obj, **kwargs)

    @pcm_method
    def bic(self, pcm, **kwargs):
        """ Map this :class:`xarray.Dataset` on :func:`pyxpcm.pcm.bic` """
        return pcm.bic(self._obj, **kwargs)

