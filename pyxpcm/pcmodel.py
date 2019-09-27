# -*- coding: utf-8 -*-
"""

.. module:: pcmodel
   :synopsis: Profile Classification Model

.. moduleauthor:: Guillaume Maze <gmaze@ifremer.fr>

Multi-variables classification, ie use of more than physical variable as PCM features

Created on 2019/09/27
@author: G. Maze (Ifremer/LOPS)
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import collections
import inspect
import dask

from scipy import interpolate
import copy
import warnings
import time
from contextlib import contextmanager

# Internal:
from .plot import _PlotMethods
# from .stats import quant as pcmstats_quant

# Scikit-learn usefull methods:
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import validation
from sklearn.exceptions import NotFittedError

####### Scikit-learn statistic backend:
# https://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# http://scikit-learn.org/stable/modules/mixture.html
from sklearn.mixture import GaussianMixture

####### Dask-ml statistic backend:
# https://ml.dask.org/modules/api.html#module-dask_ml.decomposition
# from dask_ml import preprocessing
# https://ml.dask.org/modules/generated/dask_ml.decomposition.PCA.html
# from dask_ml.decomposition import PCA



# Core PCM classes:
class PCMFeatureError(Exception):
    """Exception raised when features not found."""

# Decorators
def pcm_method(func):
    # I'd like to return pcm.<func>(self._obj, **kwargs)
    return func

@xr.register_dataset_accessor('pyxpcm')
class ds_xarray_accessor_pyXpcm:
    """
        Nothing happens in place here, so it should always goes like:

        ds = ds.pyxpcm.method()

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
                print(("Idying '%s' as '%s' in this dataset") % (feature_in_pcm, feature_in_ds))

        return feature_in_ds

    def add(self, da):
        """ Add a new xr.DataArray to the xr.DataSet """

        # Add pyXpcm tracking clues:
        da.attrs['comment'] = "Automatically added by pyXpcm"

        #
        # vname = da.name
        self._obj[da.name] = da
        return self._obj

    def clean(self):
        """ Remove from xr.dataset all variables created with pyXpcm """
        # See add() method to identify these variables.
        for vname in self._obj.data_vars:
            if ("comment" in self._obj[vname].attrs) \
                and (self._obj[vname].attrs['comment'] == "Automatically added by pyXpcm"):
                self._obj = self._obj.drop(vname)
        return self._obj

    def feature_dict(self, pcm, features=None):
        # Build the features dictionary for this dataset and PCM
        if features:
            features_dict = dict()
            for feature_in_pcm in features:
                feature_in_ds = features[feature_in_pcm]
                if not feature_in_ds:
                    feature_in_ds = self.__id_feature_name(pcm, {feature_in_pcm: None})
                features_dict[feature_in_pcm] = feature_in_ds
        else:
            features_dict = dict()
            for feature_in_pcm in pcm._props['features']:
                feature_in_ds = self.__id_feature_name(pcm, {feature_in_pcm: None})
                features_dict[feature_in_pcm] = feature_in_ds

        # Re-order the dictionary to match the PCM set order:
        for key in pcm._props['features']:
            features_dict[key] = features_dict.pop(key)

        return features_dict

    def sampling_dim(self, pcm, dim=None, features=None):
        """ Return the list of dimensions to be aggregated in sampling, for a given feature"""

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

    def mask(self, pcm, dim=None, features=None):
        """ Create a mask where PCM features are defined """
        feature_dict = self._obj.pyxpcm.feature_dict(pcm, features=features)
        SD = self.sampling_dim(pcm, dim=dim, features=features)
        M = list()
        for feature_name_in_pcm in feature_dict:
            feature_name_in_ds = feature_dict[feature_name_in_pcm]
            da = self._obj[feature_name_in_ds]

            # Is this a thick array or a slice ?
            is_slice = np.all(pcm._props['features'][feature_name_in_pcm] == None)

            if not is_slice:
                dim = SD[feature_name_in_ds]['DIM_VERTICAL']
                # z_top = np.max(pcm._props['features'][feature_name_in_pcm])
                z_bto = np.min(pcm._props['features'][feature_name_in_pcm])
                mask = self._obj[feature_name_in_ds].where(
                    self._obj[dim]>=z_bto).notnull().sum(dim=dim) == len(np.where(self._obj[dim]>=z_bto)[0])
            else:
                mask = self._obj[feature_name_in_ds].notnull()
            mask = mask.rename('PCM_MASK')
            M.append(mask)
        mask = xr.concat(M, dim='n_features')
        mask = mask.sum(dim='n_features')
        mask = mask==pcm.F
        return mask

    def quantile(self,
                  q,
                  of=None,
                  using='PCM_LABELS',
                  outname='PCM_QUANT',
                  inplace=True,
                  keep_attrs=False):

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
            ds: :class:`xarray.Dataset`
                Input dataset

            name: str, default is 'PCM_POST'
                Name of the :class:`xarray.DataArray` with prediction probability (posteriors)

            classdimname: str, default is 'pcm_class'
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

            inplace: boolean, False by default
                If False, return a :class:`xarray.DataArray` with robustness category
                If True, return the input :class:`xarray.DataSet` with robustness category added as a new :class:`xarray.DataArray`

            Returns
            -------
            :class:`xarray.DataArray`
                Robustness category of the classification

            __author__: gmaze@ifremer.fr
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
        return pcm.fit(self._obj, **kwargs)

    @pcm_method
    def predict(self, pcm, **kwargs):
        return pcm.predict(self._obj, **kwargs)

    @pcm_method
    def fit_predict(self, pcm, **kwargs):
        return pcm.fit_predict(self._obj, **kwargs)

    @pcm_method
    def predict_proba(self, pcm, **kwargs):
        return pcm.predict_proba(self._obj, **kwargs)

    @pcm_method
    def score(self, pcm, **kwargs):
        return pcm.score(self._obj, **kwargs)

    @pcm_method
    def bic(self, pcm, **kwargs):
        return pcm.bic(self._obj, **kwargs)

class NoTransform(BaseEstimator):
    """ An estimator that does nothing in fit and transform """
    def __init__(self):
        self.fitted = False

    def fit(self, *args):
        self.fitted = True
        return self

    def transform(self, x, *args):
        return x

    def score(self, x):
        return 1


class Vertical_Interpolator:
    """ Internal machinery for the interpolation of vertical profiles

        This class is called once at PCM instance initialisation
        and
        whenever data to be classified are not on the PCM feature axis.

        Here we consume numpy arrays
    """

    def __init__(self, axis=None, debug=False):
        self.axis = axis
        self._debug = debug
        # self.doINTERPz = False

    def issimilar(self, Caxis):
        """Check if the output and input axis are similar"""
        test = np.array_equiv(self.axis, Caxis)
        return test

    def isintersect(self, Caxis):
        """Check if output axis can be spot sampled from the input axis"""
        in1 = np.in1d(Caxis, self.axis)
        test = self.axis.shape[0] == np.count_nonzero(in1)
        return test

    def mix(self, x):
        """ 
            Homogeneize the upper water column: 
            Set 1st nan values to the first non-NaN value
        """
        # izmixed = np.argwhere(np.isnan(x))
        izok = np.where(~np.isnan(x))[0][0]
        # x[izmixed] = x[izok]
        x[0] = x[izok]
        return x

    def transform(self, C, Caxis):
        """
            Interpolate data on the PCM vertical axis

            C[n_samples, n_levels]

            Caxis[n_levels]

        """
        if not np.all(self.axis == None):
            if np.any(Caxis > 0):
                raise ValueError("Feature axis (depth) must be <=0 and oriented downward")
            if (self.isintersect(Caxis)):
                # Output axis is in the input axis, not need to interpolate:
                if self._debug: print(
                    "\tOutput axis is in the input axis, not need to interpolate, simple intersection")
                in1 = np.in1d(Caxis, self.axis)
                C = C[:, in1]
            elif (not self.issimilar(Caxis)):
                if self._debug: print("\tOutput axis is new, will use interpolation")
                [Np, Nz] = C.shape
                # Possibly Create a mixed layer for the interpolation to work
                # smoothly at the surface
                if ((Caxis[0] < 0.) & (self.axis[0] == 0.)):
                    Caxis = np.concatenate((np.zeros(1), Caxis))
                    x = np.empty((Np, 1))
                    x.fill(np.nan)
                    C = np.concatenate((x, C), axis=1)
                    np.apply_along_axis(self.mix, 1, C)
                    if self._debug: print("Data verticaly mixed to reached the surface")
                # Linear interpolation of profiles onto the model grid:
                f = interpolate.interp2d(Caxis, np.arange(Np), C, kind='linear')
                C = f(self.axis, np.arange(Np))

                # I don't understand why, but the interp2d return an array corresponding to a sorted Caxis.
                # Because our convention is to use a negative Caxis, oriented downward, i.e. not sorted, we
                # need to flip the interpolation results so that it oriented along Caxis and not sort(Caxis).
                if self.axis[0] > self.axis[-1]:
                    C = np.flip(C, axis=1)
            else:
                raise ValueError("I dont know how to vertically interpolate this array !")
        else:
            if self._debug: print("This is a slice array, no vertical interpolation")
            C = C  # No change
        return C


class pcm:
    """Base class for a Profile Classification Model

    Consume and return :module:`xarray` objects
    """
    def __init__(self,
                 K=1,
                 features=dict(),
                 scaling=1,
                 reduction=1, maxvar=15,
                 classif='gmm', covariance_type='full',
                 verb=False,
                 debug=False,
                 timeit=False, timeit_verb=True):
        """Create the PCM instance

        Parameters
        ----------
        K: int
            The number of class, or cluster, in the classification model.

        features: dict()
            The vertical axis to use for each features.
            eg: {'temperature':np.arange(-2000,0,1)}

        scaling: int (default: 1)
            Define the scaling method:

            - 0: No scaling
            - **1: Center on sample mean and scale by sample std**
            - 2: Center on sample mean only

        reduction: int (default: 1)
            Define the dimensionality reduction method:

            - 0: No reduction
            - **1: Reduction using :class:`sklearn.decomposition.PCA`**

        maxvar: float (default: 99.9)
            Maximum feature variance to preserve in the reduced dataset using :class:`sklearn.decomposition.PCA`. In %.

        classif: str (default: 'gmm')
            Define the classification method.
            The only method available as of now is a Gaussian Mixture Model.
            See :class:`sklearn.mixture.GaussianMixture` for more details.

        covariance_type: str (default: 'full')
            Define the type of covariance matrix shape to be used in the default classifier GMM.
            It can be ‘full’ (default), ‘tied’, ‘diag’ or ‘spherical’.

        verb: boolean (default: False)
            More verbose output

        """
        if   scaling==0: with_scaler = 'none'; with_mean=False; with_std = False
        elif scaling==1: with_scaler = 'normal'; with_mean=True; with_std = True
        elif scaling==2: with_scaler = 'center'; with_mean=True; with_std = False
        else: raise NameError('scaling must be 0, 1 or 2')
        
        if   reduction==0: with_reducer = False
        elif reduction==1: with_reducer = True
        else: raise NameError('reduction must be 0 or 1')
        
        if classif=='gmm': with_classifier = 'gmm';
        else: raise NameError("classifier must be 'gmm' (no other methods implemented at this time)")

        #todo check validity of the dict of features

        self._props = {'K': np.int(K),
                       'F': len(features),
                        'llh': None,
                        'COVARTYPE': covariance_type,
                        'with_scaler': with_scaler,
                        'with_reducer': with_reducer,
                        'with_classifier': with_classifier,
                        'maxvar': maxvar,
                        'features': collections.OrderedDict(features)}
        self._xmask = None # xarray mask for nd-array used at pre-processing steps

        self._verb = verb #todo _verb is a property, should be set/get with a decorator
        self._debug = debug

        self._interpoler = collections.OrderedDict()
        self._scaler = collections.OrderedDict()
        self._scaler_props = collections.OrderedDict()
        self._reducer = collections.OrderedDict()
        self._homogeniser = collections.OrderedDict()
        for feature_name in features:
            feature_axis = self._props['features'][feature_name]

            self._scaler[feature_name] = preprocessing.StandardScaler(with_mean=with_mean,
                                                        with_std=with_std)
            self._scaler_props[feature_name] = {'units': '?'}

            is_slice = np.all(feature_axis == None)
            if not is_slice:
                self._interpoler[feature_name] = Vertical_Interpolator(axis=feature_axis, debug=self._debug)
                if np.prod(feature_axis.shape) == 1:
                    # Single level, not need to reduce
                    if self._debug: print('Single level, not need to reduce', np.prod(feature_axis.ndim))
                    self._reducer[feature_name] = NoTransform()
                else:
                    # Multi-vertical-levels, set reducer:
                    if with_reducer:
                        self._reducer[feature_name] = PCA(n_components=self._props['maxvar'], svd_solver='full')
                    else:
                        self._reducer[feature_name] = NoTransform()
            else:
                self._interpoler[feature_name] = NoTransform()
                self._reducer[feature_name] = NoTransform()
            self._homogeniser[feature_name] = {'mean': 0, 'std': 1}

        self._classifier = GaussianMixture(n_components=self._props['K'],
                                          covariance_type=self._props['COVARTYPE'],
                                          init_params='kmeans',
                                          max_iter=1000,
                                          tol=1e-6)

        # Define the "context" to execute some functions inner code
        # (useful for time benchmarking)
        self._context = self.__empty_context # Default is empty, do nothing
        self._context_args = dict()
        if timeit:
            self._context = self.__timeit_context
            self._context_args = {'maxlevel': 3, 'verb':timeit_verb}
            self._timeit = dict()

        self._version = '0.6'

    @contextmanager
    def __timeit_context(self, name, opts=dict()):
        default_opts = {'maxlevel': np.inf, 'verb':False}
        for key in opts:
            if key in default_opts:
                default_opts[key] = opts[key]
        level = len([i for i in range(len(name)) if name.startswith('.', i)])
        if level <= default_opts['maxlevel']:
            startTime = time.time()
            yield
            elapsedTime = time.time() - startTime
            trailingspace = " " * level
            trailingspace = " "
            if default_opts['verb']:
                # print('... time in {} {}: {} ms'.format(trailingspace, name, int(elapsedTime * 1000)))
                print('{} {}: {} ms'.format(trailingspace, name, int(elapsedTime * 1000)))
            if name in self._timeit:
                self._timeit[name].append(elapsedTime * 1000)
            else:
                self._timeit[name] = list([elapsedTime*1000])
        else:
            yield

    @contextmanager
    def __empty_context(self, name, *args, **kargs):
        yield

    def __call__(self, **kwargs):
        self.__init__(**kwargs)
    
    def __iter__(self):
        self.__i = 0
        return self
    
    def __next__(self):
        if self.__i < self.K:
            i = self.__i
            self.__i += 1
            return i
        else:
            raise StopIteration()

    def __repr__(self):
        return self.display(deep=self._verb)


    def ravel(self, da, dim=None, feature_name=str()):
        """ Extract from N-d array a X(feature,sample) 2-d array and vertical dimension z

            Parameters
            ----------
            da: :class:`xarray.DataArray`
                The DataArray to process

            dim: str
                Name of the vertical dimension in the input :class:`xarray.DataArray`

            feature_name: str
                Target PCM feature name for the input :class:`xarray.DataArray`

            Returns
            -------
            X: :class:`xarray.DataArray`
                A new DataArray with dimension ['n_sampling','n_features']

            z:

            sampling_dims:

            Example
            -------
            This function is meant to be used internally only

            __author__: gmaze@ifremer.fr

        """

        # Is this a thick array or a slice ?
        is_slice = np.all(self._props['features'][feature_name] == None)

        # Load mask where all features are available for this PCM:
        mask_stacked = self._xmask

        if is_slice:
            # No vertical dimension to use, simple stacking
            sampling_dims = list(da.dims)
            # X = da.stack({'sampling': sampling_dims}).values
            # X = X[:, np.newaxis]
            # Apply all-features mask:
            X = da.stack({'sampling': sampling_dims})
            X = X.where(mask_stacked == 1, drop=True).expand_dims('dummy').transpose()#.values
            z = np.empty((1,))
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
            X = da.stack({'sampling': sampling_dims}) #todo Improve performance for this operation !
            # Apply all-features mask:
            X = X.where(mask_stacked == 1, drop=True).transpose()
            z = da[dim].values
        return X, z, sampling_dims

    def unravel(self, ds, sampling_dims, X):
        """ Create a DataArray from a numpy array and sampling dimensions """

        # Load mask where all features are available for this PCM:
        mask_stacked = self._xmask

        #
        coords = list()
        size = list()
        for dim in sampling_dims:
            coords.append(ds[dim])
            size.append(len(ds[dim]))
        da = xr.DataArray(np.empty((size)), coords=coords)
        da = da.stack({'sampling': sampling_dims})
        da = da.where(mask_stacked == 1, drop=True).transpose()
        da.values = X
        da = da.unstack('sampling')
        return da

    @property
    def K(self):
        """Return the number of classes"""
        return self._props['K']

    @property
    def F(self):
        """Return the number of features"""
        return self._props['F']

    @property
    def features(self):
        """Return the dictionnary of features"""
        return [feature for feature in self._props['features']]

    @property
    def scaler(self):
        """Return the scaler method"""
        return self._scaler

    @property
    def plot(self):
        """Access plotting functions
        """
        return _PlotMethods(self)

    @property
    def timeit(self):
        """ Return a pandas dataframe with method times """

        def get_multindex(times):
            """ Create multi-index pandas """
            # Get max levels:
            dpt = list()
            [dpt.append(len(key.split("."))) for key in times]
            max_dpt = np.max(dpt)
            # Read index:
            levels_1 = list()
            levels_2 = list()
            levels_3 = list()
            levels_4 = list()
            if max_dpt == 1:
                for key in times:
                    levels = key.split(".")
                    levels_1.append(levels[0])
                return max_dpt, [levels_1]
            elif max_dpt == 2:
                for key in times:
                    levels = key.split(".")
                    if len(levels) == 1:
                        levels_1.append(levels[0])
                        levels_2.append('total')
                    if len(levels) == 2:
                        levels_1.append(levels[0])
                        levels_2.append(levels[1])
                return max_dpt, [levels_1,levels_2]
            elif max_dpt == 3:
                for key in times:
                    levels = key.split(".")
        #             print(len(levels), levels)
                    if len(levels) == 1:
                        levels_1.append(levels[0])
                        levels_2.append('total')
                        levels_3.append('')
                    if len(levels) == 2:
                        levels_1.append(levels[0])
                        levels_2.append(levels[1])
                        levels_3.append('total')
                    if len(levels) == 3:
                        levels_1.append(levels[0])
                        levels_2.append(levels[1])
                        levels_3.append(levels[2])
                return max_dpt, [levels_1,levels_2,levels_3]
            elif max_dpt == 4:
                for key in times:
                    levels = key.split(".")
                    if len(levels) == 1:
                        levels_1.append(levels[0])
                        levels_2.append('total')
                        levels_3.append('')
                        levels_4.append('')
                    if len(levels) == 2:
                        levels_1.append(levels[0])
                        levels_2.append(levels[1])
                        levels_3.append('total')
                        levels_4.append('')
                    if len(levels) == 3:
                        levels_1.append(levels[0])
                        levels_2.append(levels[1])
                        levels_3.append(levels[2])
                        levels_4.append('total')
                    if len(levels) == 4:
                        levels_1.append(levels[0])
                        levels_2.append(levels[1])
                        levels_3.append(levels[2])
                        levels_4.append(levels[3])
                return max_dpt, [levels_1,levels_2,levels_3,levels_4]

        times = self._timeit
        max_dpt, arrays = get_multindex(times)
        if max_dpt == 1:
            index = pd.Index(arrays[0], names=['Method'])
            df = pd.Series([np.sum(times[key]) for key in times], index=index)
            # df = df.T
        elif max_dpt == 2:
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['Method', 'Sub-method'])
            df = pd.Series([np.sum(times[key]) for key in times], index=index)
            df = df.unstack(0)
            df = df.drop('total')
            df = df.T
        elif max_dpt == 3:
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['Method', 'Sub-method', 'Sub-sub-method'])
            df = pd.Series([np.sum(times[key]) for key in times], index=index)
    #         df = df.unstack(0)
        elif max_dpt == 4:
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=['Method', 'Sub-method', 'Sub-sub-method',
                                                             'Sub-sub-sub-method'])
            df = pd.Series([np.sum(times[key]) for key in times], index=index)

        return df

    def display(self, deep=False):
        """Display detailed parameters of the PCM
            This is not a get_params because it doesn't return a dictionary
            Set Boolean option 'deep' to True for all properties display
        """
        summary = [("<pcm '%s' (K: %i, F: %i)>")%(self._props['with_classifier'],
                                                  self._props['K'],
                                                  len(self._props['features']))]
        
        # PCM core properties:
        prop_info = ('Number of class: %i') % self._props['K']
        summary.append(prop_info)
        prop_info = ('Number of feature: %i') % len(self._props['features'])
        summary.append(prop_info)

        prop_info = ('Feature names: %s') % (repr(self._props['features'].keys()))
        summary.append(prop_info)

        # prop_info = ('Feature axis: [%s, ..., %s]') % (repr(self._props['features'][0]),
        #                                                repr(self._props['feature_axis'][-1]))
        # summary.append(prop_info)
        
        prop_info = ('Fitted: %r') % hasattr(self, 'fitted')
        summary.append(prop_info)

        # PCM workflow parameters:
        for feature in self._props['features']:
            prop_info = "Feature: '%s'" % feature
            summary.append(prop_info)
            summary.append("\t Interpoler: %s"%(type(self._interpoler[feature])))

            # prop_info = ('\t Sample Scaling: %r') %
            # summary.append(prop_info)
            summary.append("\t Scaler: %r, %s"%(self._props['with_scaler'], type(self._scaler[feature])))

            if (deep):
                # summary.append("\t\t Scaler properties:")
                d = self._scaler[feature].get_params(deep=deep)
                for p in d: summary.append(("\t\t %s: %r")%(p,d[p]))

            # prop_info = ('\t Dimensionality Reduction: %r') %
            # summary.append(prop_info)
            summary.append("\t Reducer: %r, %s"%(self._props['with_reducer'], type(self._reducer[feature])))

            if (deep):
                # summary.append("\t\t Reducer properties:")
                d = self._reducer[feature].get_params(deep=deep)
                for p in d: summary.append(("\t\t %s: %r")%(p,d[p]))
        # return '\n'.join(summary)

        # prop_info = ('Classification: %r') %
        # summary.append(prop_info)
        summary.append("Classifier: %r, %s"%(self._props['with_classifier'], type(self._classifier)))
        #prop_info = ('GMM covariance type: %s') % self._props['COVARTYPE']
        #summary.append(prop_info)
        if (hasattr(self,'fitted')):
            prop_info = ('\t log likelihood of the training set: %f') % self._props['llh']
            summary.append(prop_info)
        
        if (deep):
            summary.append("\t Classifier properties:")
            d = self._classifier.get_params(deep=deep)
            for p in d: summary.append(("\t\t %s: %r")%(p,d[p]))
        
        # Done
        return '\n'.join(summary)

    def copy(self):
        """Return a deep copy of the PCM instance"""
        return copy.deepcopy(self)

    def preprocessing_this(self, da, dim=None, feature_name=str(), action='?'):
        """Pre-process data before anything

        Possible pre-processing steps:

        - interpolation,
        - scaling,
        - reduction

        Parameters
        ----------
        da: :class:`xarray.DataArray`
            The DataArray to process

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataArray`

        feature_name: str
            Target PCM feature name for the input :class:`xarray.DataArray`

        Returns
        -------
        X: np.array
            Pre-processed feature, with dimensions (N_SAMPLE, N_FEATURES)

        sampling_dims: list()
            List of the input :class:`xarray.DataArray` dimensions stacked as sampling points

        """
        this_context = str(action)+'.1-preprocess.2-feature_'+feature_name
        with self._context(this_context + '.total', self._context_args):

            # MAKE THE ND-ARRAY A 2D-ARRAY
            with self._context(this_context + '.1-ravel', self._context_args):
                X, z, sampling_dims = self.ravel(da, dim=dim, feature_name=feature_name)
                if self._debug:
                    print("\tInput working arrays X (%s) and z with shapes:" % type(X), X.shape, z.shape)

            # INTERPOLATION STEP:
            with self._context(this_context + '.2-interp', self._context_args):
                X = self._interpoler[feature_name].transform(X, z)

            # FIT STEPS:
            # Based on scikit-lean methods
            # We need to fit the pre-processing methods in order to re-use them when
            # predicting a new dataset

            # SCALING:
            with self._context(this_context+'.3-scale_fit', self._context_args):
                if not hasattr(self, 'fitted'):
                    self._scaler[feature_name].fit(X.data)
                    if 'units' in da.attrs:
                        self._scaler_props[feature_name]['units'] = da.attrs['units']

            with self._context(this_context + '.4-scale_transform', self._context_args):
                X.data = self._scaler[feature_name].transform(X.data, copy=True) # Scaling

            # REDUCTION:
            with self._context(this_context + '.5-reduce_fit', self._context_args):
                if (not hasattr(self, 'fitted')) and (self._props['with_reducer']):
                    self._reducer[feature_name].fit(X)

            with self._context(this_context + '.6-reduce_transform', self._context_args):
                X = self._reducer[feature_name].transform(X) # Reduction

            if self._debug:
                print("\tPreprocessed array X (%s) with shapes:" % type(X), X.shape)

        # Output:
        return X, sampling_dims

    def preprocessing(self, ds, features=None, dim=None, action='?', mask=None):
        """ Dataset pre-processing of feature(s)

        Depending on pyXpcm set-up, pre-processing steps can be:

        - interpolation,
        - scaling,
        - reduction

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        Returns
        -------
        X: np.array
            Pre-processed set of features, with dimensions (N_SAMPLE, N_FEATURES)

        sampling_dims: list()
            List of the input :class:`xarray.DataSet` dimensions stacked as sampling points

        """
        this_context = str(action)+'.1-preprocess'
        with self._context(this_context, self._context_args):
            if self._debug:
                print("> Start preprocessing for action '%s'" % action)

            # How to we find feature variable in this dataset ?
            features_dict = ds.pyxpcm.feature_dict(self, features=features)

            # Determine the mask where all features are defined for this PCM:
            with self._context(this_context + '.1-mask', self._context_args):
                if not mask:
                    mask = ds.pyxpcm.mask(self, features=features, dim=dim)
                    # Stack all-features mask:
                    mask = mask.stack({'sampling': list(mask.dims)})
                self._xmask = mask

            # Pre-process all features and build the X array
            X = np.empty(())
            Xlabel = list() # Construct a list of string labels for each feature dimension (useful for plots)
            F = self.F # Nb of features

            for feature_in_pcm in features_dict:
                feature_in_ds = features_dict[feature_in_pcm]
                if self._debug:
                    print( ("\t> Preprocessing xarray dataset '%s' as PCM feature '%s'")%(feature_in_ds, feature_in_pcm) )

                if ('maxlevel' in self._context_args) and (self._context_args['maxlevel'] <= 2):
                    a = this_context + '.2-features'
                else:
                    a = this_context
                with self._context(a, self._context_args):
                    da = ds[feature_in_ds]
                    x, sampling_dims = self.preprocessing_this(da,
                                                               dim=dim,
                                                               feature_name=feature_in_pcm,
                                                               action=action)
                    xlabel = ["%s_%i"%(feature_in_pcm, i) for i in range(0, x.shape[1])]

                with self._context(this_context + '.3-homogeniser', self._context_args):
                    # Store full array mean and std during fit:
                    if (action == 'fit') or (action == 'fit_predict'):
                        self._homogeniser[feature_in_pcm]['mean'] = np.mean(x[:])
                        self._homogeniser[feature_in_pcm]['std'] = np.std(x[:])
                    if F>1:
                        # For more than 1 feature, we need to make them comparable,
                        # so we normalise each features by their global stats:
                        x = (x-self._homogeniser[feature_in_pcm]['mean'])/self._homogeniser[feature_in_pcm]['std']
                        if self._debug and action == 'fit':
                            print(("\tHomogenisation for fit of %s") % (feature_in_pcm))
                        elif self._debug:
                            print(("\tHomogenisation of %s using fit data") % (feature_in_pcm))
                    elif self._debug:
                        print(("\tNo need for homogenisation of %s") % (feature_in_pcm))

                if np.prod(X.shape) == 1:
                    X = x
                    Xlabel = xlabel
                else:
                    X = np.append(X, x, axis=1)
                    [Xlabel.append(i) for i in xlabel]

            with self._context(this_context + '.4-xarray', self._context_args):
                self._xlabel = Xlabel
                X = xr.DataArray(X, dims=['n_samples', 'n_features'],
                                 coords={'n_samples': range(0, X.shape[0]), 'n_features': Xlabel})

            if self._debug:
                print("> Preprocessing finished, working with final X (%s) array of shape:" % type(X), X.shape,
                      " and sampling dimensions:", sampling_dims)
        return X, sampling_dims

    def fit(self, ds, features=None, dim=None):
        """Estimate PCM parameters

        For a PCM, the fit method consists in the following operations:

        - pre-processing
            - interpolation to the ``feature_axis`` levels of the model
            - scaling
            - reduction
        - estimate classifier parameters

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        Returns
        -------
        self
        """
        with self._context('fit', self._context_args) :
            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='fit')

            # CLASSIFICATION-MODEL TRAINING:
            with self._context('fit.2-fit', self._context_args):
                self._classifier.fit(X)
            with self._context('fit.3-score', self._context_args):
                self._props['llh'] = self._classifier.score(X)
                # self._props['bic'] = self._classifier.bic(X)

        # Done:
        self.fitted = True
        return self

    def predict(self, ds, features=None, dim=None, inplace=False, name='PCM_LABELS'):
        """Predict labels for profile samples

        This method add these properties to the PCM object:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with predicted labels
            If True, return the input :class:`xarray.DataSet` with labels added as a new :class:`xarray.DataArray`

        name: str, default is 'PCM_LABELS'
            Name of the :class:`xarray.DataArray` with labels

        Returns
        -------
        :class:`xarray.DataArray`
            Component labels (if option 'inplace' = False)

        *or*

        :class:`xarray.DataSet`
            Input dataset with Component labels as a 'PCM_LABELS' new :class:`xarray.DataArray`
            (if option 'inplace' = True)
        """
        with self._context('predict', self._context_args):
            # Check if the PCM is trained:
            validation.check_is_fitted(self, 'fitted')

            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='predict')

            # CLASSIFICATION PREDICTION:
            with self._context('predict.2-predict', self._context_args):
                labels = self._classifier.predict(X)
            with self._context('predict.score', self._context_args):
                self._props['llh'] = self._classifier.score(X)

            # Create a xarray with labels output:
            with self._context('predict.3-xarray', self._context_args):
                da = self.unravel(ds, sampling_dims, labels).rename(name)
                da.attrs['long_name'] = 'PCM labels'
                da.attrs['units'] = ''
                da.attrs['valid_min'] = 0
                da.attrs['valid_max'] = self._props['K']-1
                da.attrs['llh'] = self._props['llh']

            # Add labels to the dataset:
            if inplace:
                if name in ds.data_vars:
                    warnings.warn( ("%s variable already in the dataset: overwriting")%(name) )
                return ds.pyxpcm.add(da)
            else:
                return da

    def fit_predict(self, ds, features=None, dim=None, inplace=False, name='PCM_LABELS'):
        """Estimate PCM parameters and predict classes.

        This method add these properties to the PCM object:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with predicted labels
            If True, return the input :class:`xarray.DataSet` with labels added as a new :class:`xarray.DataArray`

        name: string ('PCM_LABELS')
            Name of the DataArray holding labels.

        Returns
        -------
        :class:`xarray.DataArray`
            Component labels (if option 'inplace' = False)

        *or*

        :class:`xarray.Dataset`
            Input dataset with component labels as a 'PCM_LABELS' new :class:`xarray.DataArray` (if option 'inplace' = True)

        """
        with self._context('fit_predict', self._context_args):

            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='fit_predict')

            # CLASSIFICATION-MODEL TRAINING:
            with self._context('fit_predict.2-fit', self._context_args):
                self._classifier.fit(X)
            with self._context('fit_predict.3-score', self._context_args):
                self._props['llh'] = self._classifier.score(X)
                self._props['bic'] = self._classifier.bic(X)

            # Done:
            self.fitted = True

            # CLASSIFICATION PREDICTION:
            with self._context('fit_predict.4-predict', self._context_args):
                labels = self._classifier.predict(X)
            with self._context('fit_predict.score', self._context_args):
                self._props['llh'] = self._classifier.score(X)

            # Create a xarray with labels output:
            with self._context('fit_predict.5-xarray', self._context_args):
                da = self.unravel(ds, sampling_dims, labels).rename(name)
                da.attrs['long_name'] = 'PCM labels'
                da.attrs['units'] = ''
                da.attrs['valid_min'] = 0
                da.attrs['valid_max'] = self._props['K']-1
                da.attrs['llh'] = self._props['llh']

            # Add labels to the dataset:
            if inplace:
                if name in ds.data_vars:
                    warnings.warn( ("%s variable already in the dataset: overwriting")%(name) )
                return ds.pyxpcm.add(da)
            else:
                return da

    def predict_proba(self, ds, features=None, dim=None, inplace=False, name='PCM_POST', classdimname='pcm_class'):
        """Predict posterior probability of each components given the data

        This method adds these properties to the PCM instance:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with predicted probabilities
            If True, return the input :class:`xarray.DataSet` with probabilities added as a new :class:`xarray.DataArray`

        name: str, default is 'PCM_POST'
            Name of the DataArray with prediction probability (posteriors)

        classdimname: str, default is 'pcm_class'
            Name of the dimension holding classes

        Returns
        -------
        :class:`xarray.DataArray`
            Probability of each Gaussian (state) in the model given each
            sample (if option 'inplace' = False)

        *or*

        :class:`xarray.Dataset`
            Input dataset with Component Probability as a 'PCM_POST' new :class:`xarray.DataArray`
            (if option 'inplace' = True)


        """
        with self._context('predict_proba', self._context_args):

            # Check if the PCM is trained:
            validation.check_is_fitted(self, 'fitted')

            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='predict_proba')

            # CLASSIFICATION PREDICTION:
            with self._context('predict_proba.2-predict', self._context_args):
                post_values = self._classifier.predict_proba(X)
            with self._context('predict_proba.3-score', self._context_args):
                self._props['llh'] = self._classifier.score(X)

            # Create a xarray with posteriors:
            with self._context('predict_proba.4-xarray', self._context_args):
                P = list()
                for k in range(self.K):
                    X = post_values[:, k]
                    x = self.unravel(ds, sampling_dims, X)
                    P.append(x)
                da = xr.concat(P, dim=classdimname).rename(name)
                da.attrs['long_name'] = 'PCM posteriors'
                da.attrs['units'] = ''
                da.attrs['valid_min'] = 0
                da.attrs['valid_max'] = 1
                da.attrs['llh'] = self._props['llh']

            # Add labels to the dataset:
            if inplace:
                if name in ds.data_vars:
                    warnings.warn(("%s variable already in the dataset: overwriting") % (name))
                return ds.pyxpcm.add(da)
            else:
                return da

    def score(self, ds, features=None, dim=None):
        """Compute the per-sample average log-likelihood of the given data

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        Returns
        -------
        log_likelihood: float
            In the case of a GMM classifier, this is the Log likelihood of the Gaussian mixture given data

        """
        with self._context('score', self._context_args):

            # Check if the PCM is trained:
            validation.check_is_fitted(self, 'fitted')

            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, action='score')

            # COMPUTE THE PREDICTION SCORE:
            with self._context('score.2-score', self._context_args):
                llh = self._classifier.score(X)

        return llh

    def bic(self, ds, features=None, dim=None):
        """Compute Bayesian information criterion for the current model on the input dataset

        Only for a GMM classifier

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.DataSet`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.DataSet`

        Returns
        -------
        bic: float
            The lower the better
        """
        with self._context('bic', self._context_args):

            # Check classifier:
            if self._props['with_classifier'] != 'gmm':
                raise Exception( ("BIC is only available for the 'gmm' classifier ('%s')")%\
                                 (self._props['with_classifier']) )

            def _n_parameters(_classifier):
                """Return the number of free parameters in the model. See sklearn code"""
                _, n_features = _classifier.means_.shape
                if _classifier.covariance_type == 'full':
                    cov_params = _classifier.n_components * n_features * (n_features + 1) / 2.
                elif _classifier.covariance_type == 'diag':
                    cov_params = _classifier.n_components * n_features
                elif _classifier.covariance_type == 'tied':
                    cov_params = n_features * (n_features + 1) / 2.
                elif _classifier.covariance_type == 'spherical':
                    cov_params = _classifier.n_components
                mean_params = n_features * _classifier.n_components
                return int(cov_params + mean_params + _classifier.n_components - 1)

            # Check if the PCM is trained:
            validation.check_is_fitted(self, 'fitted')

            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, action='bic')

            # COMPUTE THE log-likelihood:
            with self._context('bic.2-score', self._context_args):
                llh = self._classifier.score(X)

            # COMPUTE BIC:
            N_samples = X.shape[0]
            bic = (-2 * llh * N_samples + _n_parameters(self._classifier) * np.log(N_samples))

        return bic