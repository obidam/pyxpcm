# -*- coding: utf-8 -*-
"""

.. module:: pcmodel
   :synopsis: Profile Classification Model

.. moduleauthor:: Guillaume Maze <gmaze@ifremer.fr>

Multi-variables classification, ie use of more than 1 feature

We need to define model feature axis for each of the feature variables to be used:
so, we can think of a dictionary like:
    m = pcm(K=K, features={'temperature':Z, 'salinity':Z, 'sea_level':None})
The idea is to have a PCM definition that does not depend on anything from a dataset.

Now for a fit, we use a specific dataset, so we need to indicate how features are to be found in the ds:
    m = m.fit(ds, features={'temperature': 'TEMP', 'salinity': 'PSAL', 'sea_level': 'SLA'}, axis='depth')
Same think for a predict of another dataset.

How to manage preprocessing with more than 1 feature ?
    preprocessing has 3 steps: interpolation, scale, reduce
Features are preprocessed in series


Created on 2019/09/27
@author: gmaze
"""

import os
import sys
import xarray as xr
import numpy as np
import collections
import inspect

from scipy import interpolate
import copy
import warnings

from .plot import _PlotMethods

from sklearn.base import BaseEstimator

from sklearn.utils import validation

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
from sklearn import preprocessing

# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# http://scikit-learn.org/stable/modules/mixture.html
from sklearn.mixture import GaussianMixture

class PCMFeatureError(Exception):
    """Exception raised when features not found."""

from sklearn.exceptions import NotFittedError

class pcm:
    """Base class for a Profile Classification Model

    Consume and return :module:`xarray` objects
    """
    def __init__(self,
                 K=1,
                 features=dict(),
                 scaling=1,
                 reduction=1, maxvar=0.999,
                 classif='gmm', covariance_type='full',
                 verb=False, debug=False):
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

        self._verb = verb #todo _verb is a property, should be set/get with a decorator
        self._debug = debug

        self._interpoler = collections.OrderedDict()
        self._scaler = collections.OrderedDict()
        self._scaler_props = collections.OrderedDict()
        self._reducer = collections.OrderedDict()
        for feature_name in features:
            feature_axis = self._props['features'][feature_name]

            self._scaler[feature_name] = preprocessing.StandardScaler(with_mean=with_mean,
                                                        with_std=with_std)
            self._scaler_props[feature_name] = {'units': '?'}

            is_slice = np.all(feature_axis == None)
            if not is_slice:
                self._interpoler[feature_name] = self.__Interp(axis=feature_axis, debug=self._debug)

                if np.prod(feature_axis.shape) == 1:
                    # Single level, not need to reduce
                    if self._debug: print('Single level, not need to reduce', np.prod(feature_axis.ndim))
                    self._reducer[feature_name] = self.__EmptyTransform()
                else:
                    # Multi-levels, set reducer:
                    self._reducer[feature_name] = PCA(n_components=self._props['maxvar'],
                                       svd_solver='full')
            else:
                self._interpoler[feature_name] = self.__EmptyTransform()
                self._reducer[feature_name] = self.__EmptyTransform()

        self._classifier = GaussianMixture(n_components=self._props['K'],
                                          covariance_type=self._props['COVARTYPE'],
                                          init_params='kmeans',
                                          max_iter=1000,
                                          tol=1e-6)
        self._version = '0.6'

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

    class __EmptyTransform(BaseEstimator):
        def __init__(self):
            self.fitted = False
            # args, _, _, values = inspect.getargvalues(inspect.currentframe())
            # values.pop("self")
            # for arg, val in values.items():
            #     setattr(self, arg, val)
        def fit(self, *args):
            self.fitted = True
            return self
        def transform(self, x, *args):
            return x
        def score(self, x):
            return 1

    class __Interp:
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
            #izmixed = np.argwhere(np.isnan(x))
            izok = np.where(~np.isnan(x))[0][0]
            #x[izmixed] = x[izok]
            x[0] = x[izok]
            return x
        
        def transform(self, C, Caxis):
            """
                Interpolate data on the PCM vertical axis
            """
            if not np.all(self.axis==None):
                if np.any(Caxis>0):
                    raise ValueError("Feature axis (depth) must be <=0")
                if (self.isintersect(Caxis)):
                    # Output axis is in the input axis, not need to interpolate:
                    if self._debug: print( "Output axis is in the input axis, not need to interpolate, simple intersection" )
                    in1 = np.in1d(Caxis, self.axis)
                    C = C[:,in1]
                elif (not self.issimilar(Caxis)):
                    if self._debug: print( "Output axis is new, will use interpolation" )
                    [Np, Nz] = C.shape
                    # Possibly Create a mixed layer for the interpolation to work
                    # smoothly at the surface
                    if ((Caxis[0]<0.) & (self.axis[0] == 0.)):
                        Caxis = np.concatenate((np.zeros(1), Caxis))
                        x = np.empty((Np,1))
                        x.fill(np.nan)
                        C = np.concatenate((x,C), axis=1)
                        np.apply_along_axis(self.mix, 1, C)
                    # Linear interpolation of profiles onto the model grid:
                    f = interpolate.interp2d(Caxis, np.arange(Np), C, kind='linear')
                    C = f(self.axis, np.arange(Np))
                else:
                    raise ValueError( "I dont know how to verticaly interpolate this array !" )
            else:
                if self._debug: print("This is a slice array, no vertical interpolation")
                C = C # No change
            return C

    @property
    def K(self):
        """Return the number of classes"""
        return self._props['K']

    @property
    def F(self):
        """Return the number of features"""
        return self._props['F']

    @property
    def scaler(self):
        """Return the scaler method"""
        return self._scaler

    @property
    def plot(self):
        """Access plotting functions
        """
        return _PlotMethods(self)

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
        prop_info = ('Number of features: %i') % len(self._props['features'])
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

    def __id_feature_name(self, ds, feature):
        """Identify the dataset variable name to be used for a given feature name

            feature must be a dictionary or None for automatic discovery
        """
        feature_name_found = False

        for feature_in_pcm in feature:
            if feature_in_pcm not in self._props['features']:
                msg = ("Feature '%s' not set in this PCM")%(feature_in_pcm)
                raise PCMFeatureError(msg)

            feature_in_ds = feature[feature_in_pcm]
            if self._debug:
                print(("Idying %s as %s in this dataset") % (feature_in_pcm, feature_in_ds))

            if feature_in_ds:
                feature_name_found = feature_in_ds in ds.data_vars

            if not feature_name_found:
                feature_name_found = feature_in_pcm in ds.data_vars
                feature_in_ds = feature_in_pcm

            if not feature_name_found:
                # Look for the feature in the dataset data variables attributes
                for v in ds.data_vars:
                    if ('feature_name' in ds[v].attrs) and (ds[v].attrs['feature_name'] is feature_in_pcm):
                        feature_in_ds = v
                        feature_name_found = True
                        continue

            if not feature_name_found:
                msg = ("Feature '%s' not found in this dataset. You may want to add the 'feature_name' "
                                  "attribute to the variable you'd like to use or provide a dictionnary")%(feature_in_pcm)
                raise PCMFeatureError(msg)
        return feature_in_ds

    def __ravel(self, da, dim=None, feature_name=str()):
        """ Extract from N-d array the X(feature,sample) 2-d and vertical dimension z"""

        # Is this a thick array or a slice ?
        is_slice = np.all(self._props['features'][feature_name] == None)

        if is_slice:
            # No vertical dimension to use, simple stacking
            sampling_dims = list(set(da.dims))
            X = da.stack({'sampling': sampling_dims}).values
            X = X[:, np.newaxis]
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

            sampling_dims = list(set(da.dims) - set([dim]))
            X = da.stack({'sampling': sampling_dims}).transpose().values
            z = da[dim].values
        return X, z, sampling_dims

    def __unravel(self, ds, sampling_dims, X):
        """ Create an DataArray from a numpy array and sampling dimensions """
        coords = list()
        size = list()
        for dim in sampling_dims:
            coords.append(ds[dim])
            size.append(len(ds[dim]))
        da = xr.DataArray(np.empty((size)), coords=coords)
        da = da.stack({'sampling': sampling_dims})
        da.values = X
        da = da.unstack('sampling')
        return da

    def preprocessing_this(self, da, dim=None, feature_name=str()):
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
        X, z, sampling_dims = self.__ravel(da, dim=dim, feature_name=feature_name)
        if self._debug:
            print('Input working arrays X and z with shapes:', X.shape, z.shape)

        # INTERPOLATION STEP:
        X = self._interpoler[feature_name].transform(X, z)

        # FIT STEPS:
        # Based on scikit-lean methods
        # We need to fit the pre-processing methods in order to re-use them when
        # predicting a new dataset
        if not hasattr(self, 'fitted'):
            # SCALING:
            self._scaler[feature_name].fit(X)
            if 'units' in da.attrs:
                self._scaler_props[feature_name]['units'] = da.attrs['units']
            # REDUCTION:
            if self._props['with_reducer']:
                self._reducer[feature_name].fit(X)

        # TRANSFORMATION STEPS:
        X = self._scaler[feature_name].transform(X) # Scaling
        X = self._reducer[feature_name].transform(X) # Reduction

        if self._debug:
            print('Preprocessed arrays X with shapes:', X.shape)

        # Output:
        return X, sampling_dims

    def preprocessing(self, ds, features=None, dim=None, action='?'):
        """Pre-process all features from a dataset

        Possible pre-processing steps:

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
        if self._debug:
            print('> Start preprocessing for %s' % action)

        if features:
            features_dict = dict()
            for feature_in_pcm in features:
                feature_in_ds = features[feature_in_pcm]
                if not feature_in_ds:
                    feature_in_ds = self.__id_feature_name(ds, {feature_in_pcm: None})
                features_dict[feature_in_pcm] = feature_in_ds

        else:
            # Build the features dictionary for this dataset:
            features_dict = dict()
            for feature_in_pcm in self._props['features']:
                feature_in_ds = self.__id_feature_name(ds, {feature_in_pcm: None})
                features_dict[feature_in_pcm] = feature_in_ds

        # Re-order the dictionary to match the PCM set order:
        for key in self._props['features']:
            features_dict[key] = features_dict.pop(key)

        if self._debug:
            print('features_dict:', features_dict)

        X = np.empty(())
        for feature_in_pcm in features_dict:
            feature_in_ds = features_dict[feature_in_pcm]
            if self._debug:
                print( ("Preprocessing %s as %s")%(feature_in_ds, feature_in_pcm) )
            da = ds[feature_in_ds]
            x, sampling_dims = self.preprocessing_this(da, dim=dim, feature_name=feature_in_pcm)
            if np.prod(X.shape) == 1:
                X = x
            else:
                X = np.append(X, x, axis=1)

        if self._debug:
            print("> Preprocessing finished, working with final X array of shape:", X.shape,
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

        # PRE-PROCESSING:
        X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='fit')

        # CLASSIFICATION-MODEL TRAINING:
        self._classifier.fit(X)
        self._props['llh'] = self._classifier.score(X)
        self._props['bic'] = self._classifier.bic(X)

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
        # Check if the PCM is trained:
        validation.check_is_fitted(self, 'fitted')

        # PRE-PROCESSING:
        X, sampling_dims = self.preprocessing(ds, features=features, action='predict')

        # CLASSIFICATION PREDICTION:
        labels = self._classifier.predict(X)
        self._props['llh'] = self._classifier.score(X)

        # Create a xarray with labels output:
        da = self.__unravel(ds, sampling_dims, labels).rename(name)
        da.attrs['long_name'] = 'PCM labels'
        da.attrs['units'] = ''
        da.attrs['valid_min'] = 0
        da.attrs['valid_max'] = self._props['K']-1
        da.attrs['llh'] = self._props['llh']

        # Add labels to the dataset:
        if inplace:
            if name in ds.data_vars:
                warnings.warn( ("%s variable already in the dataset: overwriting")%(name) )
            ds[name] = da
        else:
            return da
        #

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
        # PRE-PROCESSING:
        X, sampling_dims = self.preprocessing(ds, features=features, dim=dim, action='fit_predict')

        # CLASSIFICATION-MODEL TRAINING:
        self._classifier.fit(X)
        self._props['llh'] = self._classifier.score(X)
        self._props['bic'] = self._classifier.bic(X)

        # Done:
        self.fitted = True

        # CLASSIFICATION PREDICTION:
        labels = self._classifier.predict(X)
        self._props['llh'] = self._classifier.score(X)

        # Create a xarray with labels output:
        da = self.__unravel(ds, sampling_dims, labels).rename(name)
        da.attrs['long_name'] = 'PCM labels'
        da.attrs['units'] = ''
        da.attrs['valid_min'] = 0
        da.attrs['valid_max'] = self._props['K']-1
        da.attrs['llh'] = self._props['llh']

        # Add labels to the dataset:
        if inplace:
            if name in ds.data_vars:
                warnings.warn( ("%s variable already in the dataset: overwriting")%(name) )
            ds[name] = da
        else:
            return da
        #

    def predict_proba(self, ds, features=None, dim=None, inplace=False, name='PCM_POST', classdimname='N_CLASS'):
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

        classdimname: str, default is 'N_CLASS'
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
        # Check if the PCM is trained:
        validation.check_is_fitted(self, 'fitted')

        # PRE-PROCESSING:
        X, sampling_dims = self.preprocessing(ds, features=features, action='predict_proba')

        # CLASSIFICATION PREDICTION:
        post_values = self._classifier.predict_proba(X)
        self._props['llh'] = self._classifier.score(X)

        # Create a xarray with posteriors:
        P = list()
        for k in range(self.K):
            X = post_values[:, k]
            x = self.__unravel(ds, sampling_dims, X)
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
            ds[name] = da
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
        # Check if the PCM is trained:
        validation.check_is_fitted(self, 'fitted')

        # PRE-PROCESSING:
        X, sampling_dims = self.preprocessing(ds, features=features, action='score')

        # COMPUTE THE PREDICTION SCORE:
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
        llh = self._classifier.score(X)

        # COMPUTE BIC:
        N_samples = X.shape[0]
        bic = (-2 * llh * N_samples + _n_parameters(self._classifier) * np.log(N_samples))

        return bic