# -*- coding: utf-8 -*-
"""

.. module:: pyxpcm
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

import warnings
import time
from contextlib import contextmanager
from datetime import datetime

# Internal:
from .plot import _PlotMethods
from .stat import _StatMethods
from .utils import LogDataType, Vertical_Interpolator, NoTransform, StatisticsBackend, docstring
from . import io

# Scikit-learn useful methods:
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils import validation
from sklearn.utils import assert_all_finite
from sklearn.exceptions import NotFittedError

####### Scikit-learn statistic backend:
# http://scikit-learn.org/stable/modules/mixture.html
from sklearn.mixture import GaussianMixture

class PCMFeatureError(Exception):
    """Exception raised when features not correct"""

class PCMClassError(Exception):
    """Exception raised when classes not correct"""

class pcm(object):
    """Profile Classification Model class constructor

        Consume and return :mod:`xarray` objects

    """
    def __init__(self,
                 K:int,
                 features:dict(),
                 scaling=1,
                 reduction=1, maxvar=15,
                 classif='gmm', covariance_type='full',
                 verb=False,
                 debug=False,
                 timeit=False, timeit_verb=False,
                 chunk_size='auto',
                 backend='sklearn'):
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

        timeit: boolean (default: False)
            Register time of operation for performance evaluation

        timeit_verb: boolean (default: False)
            Print time of operation during execution

        chunk_size: 'auto' or int
            Sampling chunk size, (array of features after pre-processing)

        backend: str
            Statistic library backend, 'sklearn' (default) or 'dask_ml'

        """
        if K==0:
            raise PCMClassError("Can't create a PCM with K=0")
        if K is None:
            raise PCMClassError("K must be defined to create a PMC")
        if not bool(features):
            raise PCMFeatureError("Can't create a PCM without features")

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
                        'features': collections.OrderedDict(features),
                        'chunk_size': chunk_size,
                        'backend': backend,
                        'cmap': None}
        self._xmask = None # xarray mask for nd-array used at pre-processing steps

        self._verb = verb #todo _verb is a property, should be set/get with a decorator
        self._debug = debug

        self._interpoler = collections.OrderedDict()
        self._scaler = collections.OrderedDict()
        self._scaler_props = collections.OrderedDict()
        self._reducer = collections.OrderedDict()
        self._homogeniser = collections.OrderedDict()

        # Load estimators for a specific backend:
        bck = StatisticsBackend(backend, scaler='StandardScaler', reducer='PCA')

        for feature_name in features:
            feature_axis = self._props['features'][feature_name]
            if isinstance(feature_axis, xr.DataArray):
                self._props['features'][feature_name] = feature_axis.values

            # self._scaler[feature_name] = preprocessing.StandardScaler(with_mean=with_mean,
            #                                             with_std=with_std)
            if 'none' not in self._props['with_scaler']:
                self._scaler[feature_name] = bck.scaler(with_mean=with_mean, with_std=with_std)
            else:
                self._scaler[feature_name] = NoTransform()
            self._scaler_props[feature_name] = {'units': '?'}

            is_slice = np.all(feature_axis == None)
            if not is_slice:
                self._interpoler[feature_name] = Vertical_Interpolator(axis=feature_axis, debug=self._debug)
                if np.prod(feature_axis.shape) == 1:
                    # Single level: no need to reduce
                    if self._debug: print('Single level, not need to reduce', np.prod(feature_axis.ndim))
                    self._reducer[feature_name] = NoTransform()
                else:
                    # Multi-vertical-levels, set reducer:
                    if with_reducer:
                        self._reducer[feature_name] = bck.reducer(n_components=self._props['maxvar'],
                                                                  svd_solver='full')
                    else:
                        self._reducer[feature_name] = NoTransform()
            else:
                self._interpoler[feature_name] = NoTransform()
                self._reducer[feature_name] = NoTransform()
                if self._debug: print("%s is single level, no need to reduce" % feature_name)

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

        # Define statistics for the fit method:
        self._fit_stats = dict({'datetime': None, 'n_samples_seen_': None, 'score': None, 'etime': None})

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

    # def xmerge(self, ds, da):
    #     """ Add a new :class:`xarray.DataArray` to a :class:`xarray.Dataset` """
    #
    #     if da.name in ds.data_vars:
    #         warnings.warn(("%s variable already in the dataset: overwriting") % (da.name))
    #
    #     # Add pyXpcm tracking clues:
    #     da.attrs['comment'] = "Automatically added by pyXpcm"
    #
    #     #
    #     # vname = da.name
    #     # self._obj[da.name] = da
    #     ds = xr.merge([ds, da])
    #     return ds
    #
    # def __clean(self, ds):
    #     """ Remove all variables created with pyXpcm front a :class:`xarray.Dataset` """
    #     # See add() method to identify these variables.
    #     for vname in ds.data_vars:
    #         if ("comment" in ds[vname].attrs) \
    #             and (ds[vname].attrs['comment'] == "Automatically added by pyXpcm"):
    #             ds = ds.drop(vname)
    #     return ds

    def ravel(self, da, dim=None, feature_name=str):
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
                Note that data are always :class:`dask.array.Array`.

            z: :class:`numpy.array`
                The vertical axis of data

            sampling_dims: dict()
                Dictionary where keys are :class:`xarray.Dataset` variable names of features
                and values are another dictionary with the list of sampling dimension in
                ``DIM_SAMPLING`` key and the name of the vertical axis in the ``DIM_VERTICAL`` key.

            Examples
            --------
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

        X = X.chunk(chunks={'sampling': self._props['chunk_size']})
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

        if (np.prod(da.shape) != mask_stacked.shape[0]):
            if self._debug:
                print("\tUnravelled data not matching mask dimension, re-indexing")
            mask = mask_stacked.unstack()
            da = da.reindex_like(mask)

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
        """Return features definition dictionnary"""
        return self._props['features']

    @property
    def plot(self):
        """Access plotting functions"""
        self._plt = _PlotMethods(self)
        return self._plt

    @property
    def stat(self):
        """Access statistics functions"""
        return _StatMethods(self)

    @property
    def timeit(self):
        """ Return a :class:`pandas.DataFrame` with Execution time of method called on this instance """

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

    @property
    def backend(self):
        """Return the name of the statistic backend"""
        return self._props['backend']

    @property
    def fitstats(self):
        """ Estimator fit properties

            The number of samples processed by the estimator
            Will be reset on new calls to fit, but increments across partial_fit calls.
        """
        return self._fit_stats

    @docstring(io.to_netcdf.__doc__)
    def to_netcdf(self, ncfile, **ka):
        """ Save PCM to netcdf file

            Parameters
            ----------
            path : str
                Path to file
        """
        return io.to_netcdf(self, ncfile, **ka)

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
                    print("\t", "X RAVELED with success", str(LogDataType(X)))

            # INTERPOLATION STEP:
            with self._context(this_context + '.2-interp', self._context_args):
                X = self._interpoler[feature_name].transform(X, z)
                if self._debug:
                    if isinstance(self._interpoler[feature_name], NoTransform):
                        print("\t", "X INTERPOLATED with success (NoTransform)", str(LogDataType(X)))
                    else:
                        print("\t", "X INTERPOLATED with success", str(LogDataType(X)))
                    # print(X.values.flags['WRITEABLE'])
                    # After the interpolation step, we must not have nan in the 2d array:
                    assert_all_finite(X, allow_nan=False)

            # FIT STEPS:
            # We need to fit pre-processing methods in order to re-use them when
            # predicting a new dataset

            # SCALING:
            with self._context(this_context+'.3-scale_fit', self._context_args):
                if not hasattr(self, 'fitted'):
                    self._scaler[feature_name].fit(X.data)
                    if 'units' in da.attrs:
                        self._scaler_props[feature_name]['units'] = da.attrs['units']

            with self._context(this_context + '.4-scale_transform', self._context_args):
                try:
                    X.data = self._scaler[feature_name].transform(X.data, copy=False)
                except ValueError:
                    if self._debug: print("\t\t Fail to scale.transform without copy, fall back on copy=True")
                    try:
                        X.data = self._scaler[feature_name].transform(X.data, copy=True)
                    except ValueError:
                        if self._debug: print("\t\t Fail to scale.transform with copy, fall back on input copy")
                        X.data = self._scaler[feature_name].transform(X.data.copy())
                        pass
                    except:
                        if self._debug: print(X.values.flags['WRITEABLE'])
                        raise
                    pass
                except:
                    raise

                if self._debug:
                    print("\t", "X SCALED with success)", str(LogDataType(X)))

            # REDUCTION:
            with self._context(this_context + '.5-reduce_fit', self._context_args):
                if (not hasattr(self, 'fitted')) and (self._props['with_reducer']):

                    if self.backend == 'dask_ml':
                        # We have to convert any type of data array into a Dask array because
                        # dask_ml cannot handle anything else (!)
                        #todo Raise an issue on dask_ml github to ask why is this choice made
                        # Related issues:
                        #   https://github.com/dask/dask-ml/issues/6
                        #   https://github.com/dask/dask-ml/issues/541
                        #   https://github.com/dask/dask-ml/issues/542
                        X.data = dask.array.asarray(X.data, chunks=X.shape)

                    if isinstance(X.data, dask.array.Array):
                        self._reducer[feature_name].fit(X.data)
                    else:
                        self._reducer[feature_name].fit(X)

            with self._context(this_context + '.6-reduce_transform', self._context_args):
                X = self._reducer[feature_name].transform(X.data) # Reduction, return np.array

                # After reduction the new array is [ sampling, reduced_dim ]
                X = xr.DataArray(X,
                                 dims=['sampling', 'n_features'],
                                 coords={'sampling': range(0, X.shape[0]),
                                         'n_features': np.arange(0,X.shape[1])})
                if self._debug:
                    print("\t", "X REDUCED with success)", str(LogDataType(X)))


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
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

        Returns
        -------
        X: np.array
            Pre-processed set of features, with dimensions (N_SAMPLE, N_FEATURES)

        sampling_dims: list()
            List of the input :class:`xarray.Dataset` dimensions stacked as sampling points

        """
        this_context = str(action)+'.1-preprocess'
        with self._context(this_context, self._context_args):
            if self._debug:
                print("> Start preprocessing for action '%s'" % action)

            # How do we find feature variable in this dataset ?
            features_dict = ds.pyxpcm.feature_dict(self, features=features)

            # Determine mask where all features are defined for this PCM:
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
                    print( ("\n\t> Preprocessing xarray dataset '%s' as PCM feature '%s'")\
                           %(feature_in_ds, feature_in_pcm) )

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
                    if self._debug:
                        print("\t%s pre-processed with success, "  % feature_in_pcm, str(LogDataType(x)))

                with self._context(this_context + '.3-homogeniser', self._context_args):
                    # Store full array mean and std during fit:
                    if F>1:
                        # For more than 1 feature, we need to make them comparable,
                        # so we normalise each features by their global stats:
                        # FIT:
                        if (action == 'fit') or (action == 'fit_predict'):
                            self._homogeniser[feature_in_pcm]['mean'] = x.mean().values
                            self._homogeniser[feature_in_pcm]['std'] = x.std().values
                            #todo _homogeniser should be a proper standard scaler
                        # TRANSFORM:
                        x = (x-self._homogeniser[feature_in_pcm]['mean'])/\
                            self._homogeniser[feature_in_pcm]['std']
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
                if self._debug:
                    print("\tFeatures array shape and type for xarray:",
                          X.shape, type(X), type(X.data))
                X = xr.DataArray(X, dims=['n_samples', 'n_features'],
                                 coords={'n_samples': range(0, X.shape[0]), 'n_features': Xlabel})

            if self._debug:
                print("> Preprocessing done, working with final X (%s) array of shape:" % type(X), X.shape,
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
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

        Returns
        -------
        self
        """
        with self._context('fit', self._context_args) :
            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='fit')

            # CLASSIFICATION-MODEL TRAINING:
            with self._context('fit.fit', self._context_args):
                self._classifier.fit(X)

            with self._context('fit.score', self._context_args):
                self._props['llh'] = self._classifier.score(X)

            # Furthermore gather some information about the fit:
            self._fit_stats['score'] = self._props['llh']
            self._fit_stats['datetime'] = datetime.utcnow()
            if 'n_samples_seen_' not in self._classifier.__dict__:
                self._fit_stats['n_samples_seen_'] = X.shape[0]
            else:
                self._fit_stats['n_samples_seen_'] = self._classifier.n_samples_seen_
            if 'n_iter_' in self._classifier.__dict__:
                self._fit_stats['n_iter_'] = self._classifier.n_iter_

        # Done:
        self.fitted = True
        return self

    def predict(self, ds, features=None, dim=None, inplace=False, name='PCM_LABELS'):
        """Predict labels for profile samples

        This method add these properties to the PCM object:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with predicted labels
            If True, return the input :class:`xarray.Dataset` with labels added as a new :class:`xarray.DataArray`

        name: str, default is 'PCM_LABELS'
            Name of the :class:`xarray.DataArray` with labels

        Returns
        -------
        :class:`xarray.DataArray`
            Component labels (if option 'inplace' = False)

        *or*

        :class:`xarray.Dataset`
            Input dataset with Component labels as a 'PCM_LABELS' new :class:`xarray.DataArray`
            (if option 'inplace' = True)
        """
        with self._context('predict', self._context_args):
            # Check if the PCM is trained:
            validation.check_is_fitted(self, 'fitted')

            # PRE-PROCESSING:
            X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='predict')

            # CLASSIFICATION PREDICTION:
            with self._context('predict.predict', self._context_args):
                labels = self._classifier.predict(X)
            with self._context('predict.score', self._context_args):
                llh = self._classifier.score(X)

            # Create a xarray with labels output:
            with self._context('predict.xarray', self._context_args):
                da = self.unravel(ds, sampling_dims, labels).rename(name)
                da.attrs['long_name'] = 'PCM labels'
                da.attrs['units'] = ''
                da.attrs['valid_min'] = 0
                da.attrs['valid_max'] = self._props['K']-1
                da.attrs['llh'] = llh

            # Add labels to the dataset:
            if inplace:
                return ds.pyxpcm.add(da)
            else:
                return da

    def fit_predict(self, ds, features=None, dim=None, inplace=False, name='PCM_LABELS'):
        """Estimate PCM parameters and predict classes.

        This method add these properties to the PCM object:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with predicted labels
            If True, return the input :class:`xarray.Dataset` with labels added as a new :class:`xarray.DataArray`

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
            with self._context('fit_predict.fit', self._context_args):
                self._classifier.fit(X)
            with self._context('fit_predict.score', self._context_args):
                self._props['llh'] = self._classifier.score(X)

            # Furthermore gather some information about this fit:
            self._fit_stats['score'] = self._props['llh']
            if 'n_samples_seen_' not in self._classifier.__dict__:
                self._fit_stats['n_samples_seen_'] = X.shape[0]
            else:
                self._fit_stats['n_samples_seen_'] = self._classifier.n_samples_seen_
            if 'n_iter_' in self._classifier.__dict__:
                self._fit_stats['n_iter_'] = self._classifier.n_iter_

            # Done:
            self.fitted = True

            # CLASSIFICATION PREDICTION:
            with self._context('fit_predict.predict', self._context_args):
                labels = self._classifier.predict(X)

            # Create a xarray with labels output:
            with self._context('fit_predict.xarray', self._context_args):
                da = self.unravel(ds, sampling_dims, labels).rename(name)
                da.attrs['long_name'] = 'PCM labels'
                da.attrs['units'] = ''
                da.attrs['valid_min'] = 0
                da.attrs['valid_max'] = self._props['K']-1
                da.attrs['llh'] = self._props['llh']

            # Add labels to the dataset:
            if inplace:
                return ds.pyxpcm.add(da)
            else:
                return da

    def predict_proba(self, ds, features=None, dim=None, inplace=False, name='PCM_POST', classdimname='pcm_class'):
        """Predict posterior probability of each components given the data

        This method adds these properties to the PCM instance:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

        inplace: boolean, False by default
            If False, return a :class:`xarray.DataArray` with predicted probabilities
            If True, return the input :class:`xarray.Dataset` with probabilities added as a new :class:`xarray.DataArray`

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
            with self._context('predict_proba.predict', self._context_args):
                post_values = self._classifier.predict_proba(X)
            with self._context('predict_proba.score', self._context_args):
                self._props['llh'] = self._classifier.score(X)

            # Create a xarray with posteriors:
            with self._context('predict_proba.xarray', self._context_args):
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

            # Add posteriors to the dataset:
            if inplace:
                return ds.pyxpcm.add(da)
            else:
                return da

    def score(self, ds, features=None, dim=None):
        """Compute the per-sample average log-likelihood of the given data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

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
            with self._context('score.score', self._context_args):
                llh = self._classifier.score(X)

        return llh

    def bic(self, ds, features=None, dim=None):
        """Compute Bayesian information criterion for the current model on the input dataset

        Only for a GMM classifier

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            The dataset to work with

        features: dict()
            Definitions of PCM features in the input :class:`xarray.Dataset`.
            If not specified or set to None, features are identified using :class:`xarray.DataArray` attributes 'feature_name'.

        dim: str
            Name of the vertical dimension in the input :class:`xarray.Dataset`

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
            with self._context('bic.score', self._context_args):
                llh = self._classifier.score(X)

            # COMPUTE BIC:
            N_samples = X.shape[0]
            bic = (-2 * llh * N_samples + _n_parameters(self._classifier) * np.log(N_samples))

        return bic