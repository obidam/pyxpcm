#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 2019-09-29

import os
import sys
import numpy as np
import xarray
import xarray as xr
import dask
from sklearn.base import BaseEstimator
import importlib

def docstring(value):
    """Replace one function docstring

        To be used as a decorator
    """
    def _doc(func):
        func.__doc__ = value
        return func
    return _doc

def LogDataType(obj, prt=False):
    """ Description of data array type and shape/chunk

        Parameters
        ----------
        obj : :class:`xarray.DataArray`

        Returns
        -------
    """
    if not (isinstance(obj, np.ndarray)) \
            and not (isinstance(obj, xarray.core.dataarray.DataArray)) \
            and not (isinstance(obj, dask.array.core.Array)):
        raise ValueError("Invalid object type")

    log = list()
    log.append(type(obj))

    if isinstance(obj, xarray.core.dataarray.DataArray):
        log.append(type(obj.data))
        if (isinstance(obj.data, dask.array.core.Array)):
            log.append(obj.data.chunks)
        else:
            log.append(None)

    elif isinstance(obj, np.ndarray):
        log.append(type(obj.data))
        if (isinstance(obj.data, dask.array.core.Array)):
            log.append(obj.data.chunks)
        else:
            log.append(None)

    elif isinstance(obj, dask.array.core.Array):
        log.append(type(obj))
        log.append(obj.chunks)

    if prt:
        print(log)
    return log


class NoTransform(BaseEstimator):
    """ An estimator that does nothing in fit and transform """
    def __init__(self):
        self.fitted = False

    def fit(self, *args):
        self.fitted = True
        return self

    def transform(self, x, *args, **kwargs):
        return x

    def score(self, x):
        return 1


class Vertical_Interpolator(object):
    """ Internal machinery for the interpolation of vertical profiles

        This class is called once at PCM instance initialisation
        and
        whenever data to be classified are not on the PCM feature axis.

        Here we consume numpy arrays
    """

    def __init__(self, axis=None, debug=False):
        self.axis = axis
        self._debug = debug

    def issimilar(self, Caxis):
        """Check if the output and input axis are similar"""
        test = np.array_equiv(self.axis, Caxis)
        return test

    def isintersect(self, Caxis):
        """Check if output axis can be spot sampled from the input axis"""
        in1 = np.in1d(Caxis, self.axis)
        test = self.axis.shape[0] == np.count_nonzero(in1)
        return test

    def mix_deprec(self, x):
        """
            Homogeneize the upper water column:
            Set 1st nan values to the first non-NaN value
        """
        # izmixed = np.argwhere(np.isnan(x))
        izok = np.where(~np.isnan(x))[0][0]
        # x[izmixed] = x[izok]
        x[0] = x[izok]
        return x

    def mix(self, C, Caxis, vertical_dim):
        """
            Homogeneize the upper water column:
            Set 1st nan values to the first non-NaN value
        """
        if self._debug: print("\t\tData array before vertical mixing: %s" % str(LogDataType(C)))

        [Np, Nz] = C.shape

        # If data starts below the surface and feature axis requested is at the surface,
        # we add one surface level to data, and "mix" it:
        Caxis = np.concatenate((np.zeros(1), Caxis))
        x = np.empty((Np, 1))
        x.fill(np.nan)
        x = xr.DataArray(x, dims=['sampling', vertical_dim],
                         coords={'sampling': C['sampling'], vertical_dim: np.zeros((1,))})
        C = xr.concat((x, C), dim=vertical_dim)
        # Fill in to the surface the 1st non-nan value (same as self.mix, but optimized)

        # Need to compute(), otherwise bfill is not applied:
        # https://github.com/pydata/xarray/issues/2699
        C = C.compute()

        C = C.bfill(dim=vertical_dim)  # backward filling, i.e. ocean upward
        Caxis = C[vertical_dim]

        # if (isinstance(C.data, dask.array.core.Array)):
        # The concat operation above adds a chunk to the vertical dimension:
        # we go from:
        #   ((n_samples,), (n_levels,))
        # to:
        #   ((n_samples,), (1, n_levels))
        # so we need to fix the chunking scheme back to a
        # single vertical chunck, otherwise transform/interpolation will fail:
        C = C.chunk(chunks={vertical_dim: len(C[vertical_dim])})

        if self._debug: print("\t\tData array vertically mixed to reached the surface: %s" % str(LogDataType(C)))

        return C, Caxis

    def transform(self, C, Caxis):
        """
            Interpolate data on a PCM vertical axis

            C[n_samples, n_levels]

            Caxis[n_levels]
        """

        # Sanity checks:
        if not isinstance(C, xr.DataArray):
            raise ValueError("Transform works with xarray.DataArray only")
        elif 'sampling' not in C.dims:
            raise KeyError("Transform only works with xarray.DataArray with a 'sampling' dimension")
        if np.min(Caxis)>np.min(self.axis):
            raise ValueError("xarray.DataArray vertical axis is not deep enough for this PCM axis [%0.2f > %0.2f]"
                             %(np.min(Caxis), np.min(self.axis)))

        vertical_dim = list(C.dims)
        vertical_dim.remove('sampling')  # Because C must be an output of pcm.ravel()
        vertical_dim = vertical_dim[0]

        if not np.all(self.axis == None):
            if np.any(Caxis > 0):
                raise ValueError("Feature axis (depth) must be <=0")

            # todo Check if feature axis is oriented downward

            if (self.isintersect(Caxis)):
                # Output axis is in the input axis, not need to interpolate:
                if self._debug: print(
                    "\t\tOutput axis is in the input axis, not need to interpolate, simple intersection")
                in1 = np.in1d(Caxis, self.axis)
                C = C[{vertical_dim: in1}]

            elif (not self.issimilar(Caxis)):
                if self._debug: print("\t\tOutput axis is new, will use interpolation")
                # [Np, Nz] = C.shape

                ##########################
                # Possibly create a mixed layer for the interpolation to work smoothly at the surface:
                if ((Caxis[0] < 0.) & (self.axis[0] == 0.)):
                    C, Caxis = self.mix(C, Caxis, vertical_dim)

                ##########################
                # Linear interpolation of profiles onto the model grid:
                if self._debug: print("\t\tData array before interpolation: %s" % (str(LogDataType(C))))

                C = C.interp(coords={vertical_dim: self.axis})

                if self._debug: print("\t\tData array after interpolation: %s" % (str(LogDataType(C))))

                # I don't understand why, but the interp2d return an array corresponding to a sorted Caxis.
                # Because our convention is to use a negative Caxis, oriented downward, i.e. not sorted, we
                # need to flip the interpolation results so that it's oriented along Caxis and not sort(Caxis).
            #                 if self.axis[0] > self.axis[-1]:
            #                     C = np.flip(C, axis=1)

            else:
                raise ValueError("I dont know how to vertically interpolate this array !")

        elif self._debug:
            print("\t\tThis is a slice array, no vertical interpolation")
        return C


class StatisticsBackend(object):
    """ Try to implement a flexible way of changing the statistic library used by pyXpcm for scaling and reduction methods

        Users can decide if they want to use sklearn or dask_ml or any other statistic library

    """

    def __init__(self, backend='sklearn', scaler='StandardScaler', reducer='PCA'):
        """ Create a Statistic Backend

            Use classic backends:
                bck = StatisticsBackend('sklearn') # Default
                bck = StatisticsBackend('dask_ml')

            Use your own packages and specify estimator's name:
                bck = StatisticsBackend('myown_statistic_module', scaler='myscaler_method')

            Use no package but directly your predictors:
            class my_scaler():
                def __init__(self,offset):
                    self.offset = offset
                def fit(self, x):
                    return self
                def transform(self, x):
                    return x + self.offset
            bck = StatisticsBackend('', scaler=my_scaler)

        """
        self.backend = backend
        self.scaler_method = scaler
        self.reducer_method = reducer

        backends = ['sklearn', 'dask_ml']
        if (backend in backends):
            self.backend_type = 'classic'
            try:
                importlib.import_module(backend)
            except ModuleNotFoundError:
                raise ValueError("This statistic backend is not available (%s)" % backend)

            scalers = ['StandardScaler', 'MinMaxScaler', 'RobustScaler']
            if scaler not in scalers:
                raise ValueError(
                    "unrecognized scaler for StatisticsBackend: {}\n"
                    "must be one of: {}".format(scaler, scalers)
                )

            reducers = ['PCA']
            if reducer not in reducers:
                raise ValueError(
                    "unrecognized reducer for StatisticsBackend: {}\n"
                    "must be one of: {}".format(reducer, reducers)
                )

        elif len(backend) > 0:
            self.backend_type = 'custom'
            try:
                importlib.import_module(backend)
            except ModuleNotFoundError:
                raise ValueError("This statistic backend is not available (%s)" % backend)
            except ValueError:
                raise ValueError("One of the method is not a proper estimator")
            except:
                raise
        else:
            self.backend_type = 'inline'
            self.__check_estimator(scaler)
            self.__check_estimator(reducer)

    def __check_estimator(self, method):
        """ Basic check on method

            Not as complete as: sklearn.utils.estimator_checks.check_estimator
        """
        if not callable(getattr(method, "fit", None)):
            raise ValueError("Not a proper estimator (no 'fit' method)")
        if not callable(getattr(method, "transform", None)):
            raise ValueError("Not a proper estimator (no 'transform' method)")
        return method

    def scaler(self, **kwargs):
        if self.backend == 'sklearn':
            s = importlib.import_module(self.backend)
            method = getattr(getattr(s, 'preprocessing'), self.scaler_method)
        elif self.backend == 'dask_ml':
            s = importlib.import_module('dask_ml.preprocessing')
            method = getattr(s, self.scaler_method)
        elif self.backend_type == 'custom':
            s = importlib.import_module(self.backend)
            method = getattr(s, self.scaler_method)
        elif self.backend_type == 'inline':
            method = self.scaler_method
        return self.__check_estimator(method)(**kwargs)

    def reducer(self, **kwargs):
        if self.backend == 'sklearn':
            s = importlib.import_module(self.backend)
            method = getattr(getattr(s, 'decomposition'), self.reducer_method)
        elif self.backend == 'dask_ml':
            s = importlib.import_module('dask_ml.decomposition')
            method = getattr(s, self.reducer_method)
        elif self.backend_type == 'custom':
            s = importlib.import_module(self.backend)
            method = getattr(s, self.reducer_method)
        elif self.backend_type == 'inline':
            method = self.reducer_method
        return self.__check_estimator(method)(**kwargs)

