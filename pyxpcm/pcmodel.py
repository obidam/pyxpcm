# -*- coding: utf-8 -*-
"""

.. module:: pcmodel
   :synopsis: Profile Classification Model

.. moduleauthor:: Guillaume Maze <gmaze@ifremer.fr>

Consume xarray.datasets only
Variables in the dataset used for the classification are assumed to already be in 2d: [n_samples, n_features]

USE CASE:

K sets the nb of classes
Z sets the vertical axis of the model:
    K = 8
    Z = np.arange(0,1000,5)
Then we create the PCM instance:
    m = pcm(K=K, feature_axis=Z, feature_name='temperature')
feature_axis defines the model vertical axis on which data must be defined or interpolated
feature_name defines the variable name to be used from the xr.dataset for classification

Now, open a dataset already in 2d: [n_samples, n_features]:
    ds = xr.open_dataset('PROFILES.nc')

And fit a PCM:
    m = m.fit(ds)
We don't need to provide the name of the feature axis in ds (that would be 'DEPTH') to be compared against the PCM
own axis 'feature axis' because in this version, we assume all features to be already in
2d-like: [n_samples, n_features]. So we know by default that we need to compare the 2nd dimension axis of the feature
to the PCM own feature_axis array.
    ds['temperature'].dims[1] to be compared with m.feature_axis


Note that if the m.feature_name string property (eg: 'temperature') is not a variable in
the dataset, we may want to look for an attributes 'feature_name' to identify it.
    ds['TEMP'].attrs['feature_name'] = 'temperature'
This allows to classify a dataset without changing variables name by simply adding an attribute to one of them

We can also provide a dictionary to the fit/predict methods
    m = m.fit(ds, features={'temperature':'TEMP'})

If we anticipate on the multi-variables classification,
we will need to define model feature axis for each of the feature variables to be used,
so, we can think of a dictionary like:
    m = pcm(K=K, features={'temperature':Z, 'salinity':Z, 'sea_level':None})

If, again, we assume the dataset is already prepared with 2-d like [n_samples, n_features] variables, we simply need to
map dataset variable names to the PCM feature names
    m = m.fit(ds, feature={'temperature': 'TEMP', 'salinity': 'PSAL', 'sea_level': 'SLA'})


todo: Add the method to compute the appropriate BIC with independant samples

Created on 2017/09/26
@author: gmaze
"""

import os
import sys
import xarray as xr
import numpy as np

from scipy import interpolate
import copy
import warnings

from .plot import _PlotMethods

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
                 feature_axis=None,
                 feature_name=None,
                 scaling=1,
                 reduction=1, maxvar=99.9,
                 classif='gmm', covariance_type='full',
                 verb=False):
        """Create the PCM instance

        Parameters
        ----------
        K: int
            The number of class, or cluster, in the classification model.

        feature_axis: :class:`numpy.array`
            The vertical axis that the PCM will work with (to train the classifier or to classify new data).
        feature_name: str
            The name to be used when searching for the feature variable in a :class:`xarray.Dataset`.
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
        
        self._props = {'K': np.int(K),
                        'llh': None,
                        'COVARTYPE': covariance_type,
                        'with_scaler': with_scaler,
                        'with_reducer': with_reducer,
                        'with_classifier': with_classifier,
                        'maxvar': maxvar,
                        'feature_axis': np.float32(feature_axis),
                        'feature_name': feature_name}
        self._verb = verb #todo _verb is a property, should be set/get with a decorator
        
        self._interpoler = self.__Interp(axis=self._props['feature_axis'], verb=self._verb)
        
        self._scaler = preprocessing.StandardScaler(with_mean=with_mean,
                                                    with_std=with_std)
        self._scaler_props = {'units': '?'}
        self._reducer = PCA(n_components=self._props['maxvar']/100,
                           svd_solver='full')
        self._classifier = GaussianMixture(n_components=self._props['K'],
                                          covariance_type=self._props['COVARTYPE'],
                                          init_params='kmeans',
                                          max_iter=1000,
                                          tol=1e-6)
        self._version = '0.5'

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

    class __Interp:
        """ Internal machinery for the interpolation of vertical profiles
            
            This class is called once at PCM instance initialisation
            and
            whenever data to be classified are not on the PCM feature axis.

            Here we consume numpy arrays
        """
        def __init__(self, axis=None, verb=False):
            self.axis = axis
            self.verb = verb
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
            if np.any(Caxis>0):
                raise ValueError("Feature axis (depth) must be <=0")
            if (self.isintersect(Caxis)):
                # Output axis is in the input axis, not need to interpolate:
                if self.verb: print( "Output axis is in the input axis, not need to interpolate, simple intersection" )
                in1 = np.in1d(Caxis, self.axis)
                C = C[:,in1]
            elif (not self.issimilar(Caxis)):
                if self.verb: print( "Output axis is new, will use interpolation" )
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
                warnings.warn( "This is new situation I don't like !" )
            return C

    @property
    def K(self):
        """Return the number of classes"""
        return self._props['K']

    @property
    def feature_axis(self):
        """Return the feature axis array"""
        return self._props['feature_axis']

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
        """Display detailled parameters of the PCM
            This is not a get_params because it doesn't return a dictionnary
            Set Boolean option 'deep' to True for all properties display
        """
        summary = [("<pcm '%s' (K: %i, Z: %i)>")%(self._props['with_classifier'],self._props['K'],self._props['feature_axis'].size)]
        
        # PCM core properties:
        prop_info = ('Number of class: %i') % self._props['K']
        summary.append(prop_info)

        prop_info = ('Feature name: %s') % (repr(self._props['feature_name']))
        summary.append(prop_info)

        prop_info = ('Feature axis: [%s, ..., %s]') % (repr(self._props['feature_axis'][0]),repr(self._props['feature_axis'][-1]))
        summary.append(prop_info)
        
        prop_info = ('Fitted: %r') % hasattr(self,'fitted')
        summary.append(prop_info)
        
        # PCM workflow parameters:
        prop_info = ('Feature axis Interpolation:')
        summary.append(prop_info)    
        summary.append("\t Interpoler: %s"%(type(self._interpoler)))
        
        prop_info = ('Sample Scaling: %r') % self._props['with_scaler']
        summary.append(prop_info)
        summary.append("\t Scaler: %s"%(type(self._scaler)))
        
        if (deep):
            summary.append("\t Scaler properties:")
            d = self._scaler.get_params(deep=deep)
            for p in d: summary.append(("\t\t %s: %r")%(p,d[p]))
        
        prop_info = ('Dimensionality Reduction: %r') % self._props['with_reducer']
        summary.append(prop_info)       
        summary.append("\t Reducer: %s"%(type(self._reducer)))
        
        if (deep):
            summary.append("\t Reducer properties:")
            d = self._reducer.get_params(deep=deep)
            for p in d: summary.append(("\t\t %s: %r")%(p,d[p]))
        
        prop_info = ('Classification: %r') % self._props['with_classifier']
        summary.append(prop_info) 
        summary.append("\t Classifier: %s"%(type(self._classifier)))
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
    
    def __repr__(self):
        return self.display(deep=self._verb)
    
    def copy(self):
        """Return a deep copy of the PCM instance"""
        return copy.deepcopy(self)

    def __id_feature_name(self, ds, feature):
        """Identify the dataset variable name to be used as a feature
            feature must be dictionary or None for automatic discovery
        """
        if isinstance(feature, dict):
            if self._props['feature_name'] in feature:
                real_feature_name = feature[self._props['feature_name']]
                feature_name_found = real_feature_name in ds.data_vars
            else:
                raise PCMFeatureError( ("%s not found in this dictionnary")%(self._props['feature_name']) )
        elif feature == None:
            # Look for the feature in variables:
            real_feature_name = self._props['feature_name']
            feature_name_found = real_feature_name in ds.data_vars
            # or in variable attributes:
            if not feature_name_found:
                for v in ds.data_vars:
                    if ('feature_name' in ds[v].attrs) and (ds[v].attrs['feature_name'] is real_feature_name):
                        real_feature_name = v
                        feature_name_found = True
                        continue
        else:
            feature_name_found = False
        if not feature_name_found:
            # raise Exception( ("Feature '%s' not found in this dataset. You may want to add the 'feature_name' "
            #                   "attribute to the variable you'd like to use or provide a dictionnary")%(self._props['feature_name']) )
            msg = ("Feature '%s' not found in this dataset. You may want to add the 'feature_name' "
                              "attribute to the variable you'd like to use or provide a dictionnary")%(self._props['feature_name'])
            # raise(msg)
            raise PCMFeatureError(msg)
        return real_feature_name

    def preprocessing(self, ds, feature=None):
        """Pre-process data before anything

        Possible pre-processing steps:

        - interpolation,
        - scaling,
        - reduction

        Parameters
        ----------
        ds: :class:`xarray.DataSet`
        feature (optional): the :class:`xarray.DataSet` variable name to be used as the PCM feature. If not specified, the
            variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        Returns
        -------
        X, the pre-processed feature variable

        Example
        -------
        m = pcm(K=8, feature_axis=range(0,-1000,-10), feature_name='temperature')
        X = m.preprocessing(ds)
        X = m.preprocessing(ds, feature={'temperature':'TEMP'})
            
        """
        # Identify feature dims in this dataset:
        feature_name = self.__id_feature_name(ds, feature)
        sampling_axis_name = str(ds[feature_name].dims[0])
        feature_axis_name = str(ds[feature_name].dims[1])
        #todo Implement check on the implicit assumption that dimensions are sampling/feature as dim(0) and dim(1)
        #in the mean time we simply raise a warning if the sampling axis is smaller than the feature one:
        if len(ds[feature_name][feature_axis_name]) > len(ds[feature_name][sampling_axis_name]):
            warnings.warn(("More variables (%s: %i) than samples (%s: %i)") % \
                          (feature_axis_name, len(ds[feature_name][feature_axis_name]),\
                           sampling_axis_name, len(ds[feature_name][sampling_axis_name])))
        if len(ds[feature_name].dims)>2:
            raise ValueError( ("pyXpcm does not handle N-dimensional arrays yet (%i), please use 2-D arrays only")%\
                              (len(ds[feature_name].dims)))

        # Ensure that feature is of shape [n_samples, n_features]:
        #   - the 2nd dimension of the feature_name variable must be the
        #       specified feature_axis
        # if (feature_axis != None):
        #     if not ds[feature_name].dims[1] == feature_axis:
        #         raise ValueError
        # else:
        #     feature_axis = str(ds[feature_name].dims[1])

        # INTERPOLATION STEP:
        # (in this version, the interpoler class keeps consuming numpy arrays)
        X = self._interpoler.transform(ds[feature_name].values,
                                           ds[feature_name][feature_axis_name].values)

        # FIT STEPS:
        # Based on scikit-lean methods
        # We need to fit the pre-processing methods in order to re-use them when
        # predicting a new dataset
        if not hasattr(self,'fitted'):
            # SCALING:
            self._scaler.fit(X)
            if 'units' in ds[feature_name].attrs:
                self._scaler_props['units'] = ds[feature_name].attrs['units']
            # REDUCTION:
            if self._props['with_reducer']:
                self._reducer.fit(X)

        # TRANSFORMATION STEPS:
        X = self._scaler.transform(X) # Scaling
        X = self._reducer.transform(X) # Reduction

        # Output:
        return X

    def fit(self, ds, feature=None):
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

        feature: str
            The variable name in the :class:`xarray.DataSet` to be used as PCM feature. If not specified, the
            variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        Returns
        -------
        self
        """
        # Identify the feature name in the dataset:
        # feature_name = self.__id_feature_name(ds)

        # PRE-PROCESSING:
        X = self.preprocessing(ds, feature=feature)

        # CLASSIFICATION-MODEL TRAINING:
        self._classifier.fit(X)
        self._props['llh'] = self._classifier.score(X)
        self._props['bic'] = self._classifier.bic(X)

        # Done:
        self.fitted = True
        return self

    def predict(self, ds, feature=None, inplace=False, name='PCM_LABELS'):
        """Predict labels for profile samples

        This method add these properties to the PCM object:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.DataSet`

        feature: str, optional
            The :class:`xarray.DataSet` variable name to be used as the PCM feature. If not specified, the
            variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        inplace: boolean, False by default
            If False, the method returns a :class:`xarray.DataArray` with predicted labels
            If True, a :class:`xarray.DataArray` with labels is added to the input :class:``xarray.DataSet``

        name: str, default is 'PCM_LABELS'
            Name of the DataArray with labels

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
        X = self.preprocessing(ds, feature=feature)

        # CLASSIFICATION PREDICTION:
        labels = self._classifier.predict(X)
        self._props['llh'] = self._classifier.score(X)

        # Prepare xarray for output:

        # Identify the feature name in the dataset:
        feature_name = self.__id_feature_name(ds, feature=feature)
        sampling_axis_name = str(ds[feature_name].dims[0])
        # feature_axis_name = str(ds[feature_name].dims[1])

        # Identify the sampling dimension name:
        # dim_sample_name = list(set(ds[feature_name].dims).symmetric_difference([feature_axis]))[0]

        # Create a xarray.DataArray with labels:
        # labels = xr.DataArray(labels, dims=sampling_axis_name, name=name)
        labels = xr.DataArray(labels, dims=sampling_axis_name, coords={sampling_axis_name: ds[sampling_axis_name]}, name=name)
        labels.attrs['long_name'] = 'PCM labels'
        labels.attrs['units'] = ''
        labels.attrs['valid_min'] = 0
        labels.attrs['valid_max'] = self._props['K']-1
        labels.attrs['llh'] = self._props['llh']

        # Add labels to the dataset:
        if inplace:
            if name in ds.data_vars:
                warnings.warn( ("%s variable already in the dataset: overwriting")%(name) )
            ds[name] = labels
        else:
            return labels
        #

    def fit_predict(self, ds, feature=None, inplace=False, name='PCM_LABELS'):
        """Estimate PCM parameters and predict classes.

        This method add these properties to the PCM object:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`
            The dataset to work with

        feature: str
            The :class:`xarray.Dataset` variable name to be used as the PCM feature. If not specified, the variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        inplace: boolean (False by default)
            If False, the method returns a :class:`xarray.DataArray` with predicted labels
            If True, a :class:`xarray.DataArray` with labels is added to the input :class:`xarray.Dataset`

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
        X = self.preprocessing(ds, feature=feature)

        # CLASSIFICATION-MODEL TRAINING:
        self._classifier.fit(X)
        self._props['llh'] = self._classifier.score(X)

        # Done fitting
        self.fitted = True

        # CLASSIFICATION PREDICTION:
        labels = self._classifier.predict(X)
        self._props['llh'] = self._classifier.score(X)

        # Now prepare xarray for output:

        # Identify the feature name in the dataset:
        feature_name = self.__id_feature_name(ds, feature=feature)
        sampling_axis_name = str(ds[feature_name].dims[0])
        # feature_axis_name = str(ds[feature_name].dims[1])

        # Identify the sampling dimension name:
        # dim_sample_name = list(set(ds[feature_name].dims).symmetric_difference([feature_axis]))[0]

        # Create a xarray.DataArray with labels:
        # labels = xr.DataArray(labels, dims=sampling_axis_name, name=name)
        labels = xr.DataArray(labels, dims=sampling_axis_name, coords={sampling_axis_name: ds[sampling_axis_name]}, name=name)
        labels.attrs['long_name'] = 'PCM labels'
        labels.attrs['units'] = ''
        labels.attrs['valid_min'] = 0
        labels.attrs['valid_max'] = self._props['K']-1
        labels.attrs['llh'] = self._props['llh']

        # Add labels to the dataset:
        if inplace:
            if name in ds.data_vars:
                warnings.warn( ("%s variable already in the dataset: overwriting")%(name) )
            ds[name] = labels
        else:
            return labels
        #

    def predict_proba(self, ds, feature=None, inplace=False, name='PCM_POST', classdimname='N_CLASS'):
        """Predict posterior probability of each component given the data

        This method adds these properties to the PCM instance:

        - ``llh``: The log likelihood of the model with regard to new data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`

        feature: str, optional
            The :class:`xarray.Dataset` variable name to be used as the PCM feature. If not specified, the
            variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        inplace: boolean, False by default
            If False, the method returns a :class:`xarray.DataArray` with predicted labels
            If True, a :class:`xarray.DataArray` with labels is added to the input :class:`xarray.DataArray`

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
        X = self.preprocessing(ds, feature=feature)

        # CLASSIFICATION PREDICTION:
        post_values = self._classifier.predict_proba(X)
        self._props['llh'] = self._classifier.score(X)

        # Prepare xarray output:

        # Identify the feature name in the dataset:
        feature_name = self.__id_feature_name(ds, feature=feature)
        sampling_axis_name = str(ds[feature_name].dims[0])

        # Create a xarray.DataArray with posteriors:
        # labels = xr.DataArray(labels, dims=sampling_axis_name, coords={sampling_axis_name: ds[sampling_axis_name]}, name=name)
        # post = xr.DataArray(post_values, coords=[
        #                 (sampling_axis_name, ds[sampling_axis_name]),
        #                 (classdimname, range(0, self._props['K']))],
        #                     name=name)
        post = xr.DataArray(post_values,\
                            dims=[sampling_axis_name, classdimname],\
                            coords={
                        sampling_axis_name: ds[sampling_axis_name],\
                        classdimname: range(0, self._props['K'])},
                            name=name)
        post.attrs['long_name'] = 'PCM posteriors'
        post.attrs['units'] = ''
        post.attrs['valid_min'] = 0
        post.attrs['valid_max'] = 1
        post.attrs['llh'] = self._props['llh']

        # Add labels to the dataset:
        if inplace:
            if name in ds.data_vars:
                warnings.warn(("%s variable already in the dataset: overwriting") % (name))
            ds[name] = post
        else:
            return post

    def score(self, ds, feature=None):
        """Compute the per-sample average log-likelihood of the given data

        Parameters
        ----------
        ds: :class:`xarray.Dataset`

        feature: str, optional
            The :class:`xarray.Dataset` variable name to be used as the PCM feature. If not specified, the
            variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        Returns
        -------
        log_likelihood: float
            In the case of a GMM classifier, this is the Log likelihood of the Gaussian mixture given data

        """
        # Check if the PCM is trained:
        validation.check_is_fitted(self, 'fitted')

        # PRE-PROCESSING:
        X = self.preprocessing(ds, feature=feature)

        # COMPUTE THE PREDICTION SCORE:
        llh = self._classifier.score(X)

        return llh

    def bic(self, ds, feature=None):
        """Compute Bayesian information criterion for the current model on the input dataset

        Only for a GMM classifier

        Parameters
        ----------
        ds: :class:`xarray.Dataset`

        feature: str, optional
            The :class:`xarray.Dataset` variable name to be used as the PCM feature. If not specified, the
            variable is identified as PCM['feature_name'] or the variable having it as an attribute.

        Returns
        -------
        bic: float
            The lower the better
        """

        # Check classifier:
        if self._props['with_classifier'] != 'gmm':
            raise Exception( ("BIC is only available for 'gmm' classifier ('%s')")%\
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
        X = self.preprocessing(ds, feature=feature)

        # COMPUTE THE log-likelihood:
        llh = self._classifier.score(X)

        # COMPUTE BIC:
        N_samples = X.shape[0]
        bic = (-2 * llh * N_samples + _n_parameters(self._classifier) * np.log(N_samples))

        return bic