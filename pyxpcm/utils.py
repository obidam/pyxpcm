#!/bin/env python
# -*coding: UTF-8 -*-
#
# HELP
#
# Created by gmaze on 2019-09-29

import os
import sys
import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator
# from scipy import interpolate

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

class Vertical_Interpolator_deprec:
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
                raise ValueError("Feature axis (depth) must be <=0")

            #todo Check if feature axis is oriented downward

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
                    # If data starts below the surface and feature axis requested is at the surface,
                    # we add one surface level to data, and mix it
                    Caxis = np.concatenate((np.zeros(1), Caxis))
                    x = np.empty((Np, 1))
                    x.fill(np.nan)
                    # if self._debug: print("\tData type before concatenate: %s" % type(C))

                    if isinstance(C, xr.DataArray):
                        other_dim = list(C.dims)
                        other_dim.remove('sampling') # Because C must be an output of pcm.ravel()
                        other_dim = other_dim[0]
                        surface_value = xr.DataArray(x,
                                                     dims=['sampling', other_dim],
                                                     coords={'sampling': C['sampling'], other_dim: np.zeros((1,))})
                        C = xr.concat((surface_value, C), dim=other_dim)
                    else:
                        C = np.concatenate((x, C), axis=1) # This is where we go from xr.DataArray to np.ndarray
                    # if self._debug: print("\tData type after concatenate: %s" % type(C))

                    # if self._debug: print("\tData type before apply_along_axis: %s" % type(C))
                    np.apply_along_axis(self.mix, 1, C)
                    # if self._debug: print("\tData type after apply_along_axis: %s" % type(C))

                    if self._debug:
                        print("\tData (%s) vertically mixed to reached the surface" % type(C))

                # Linear interpolation of profiles onto the model grid:
                # if self._debug: print("\tData type before interpolation: %s" % type(C))
                f = interpolate.interp2d(Caxis, np.arange(Np), C.data, kind='linear')
                Ci = f(self.axis, np.arange(Np))
                # if self._debug: print("\tData type after interpolation: %s" % type(C))

                # I don't understand why, but the interp2d return an array corresponding to a sorted Caxis.
                # Because our convention is to use a negative Caxis, oriented downward, i.e. not sorted, we
                # need to flip the interpolation results so that it oriented along Caxis and not sort(Caxis).
                if self.axis[0] > self.axis[-1]:
                    C = np.flip(C, axis=1)

            else:
                raise ValueError("I dont know how to vertically interpolate this array !")

        elif self._debug:
            print("\tThis is a slice array, no vertical interpolation")
        return C

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
        if not isinstance(C, xr.DataArray):
            raise ValueError("Transform works with xarray.DataArray only")
        elif 'sampling' not in C.dims:
            raise KeyError("Transform only works with xarray.DataArray with a 'sampling' dimension")

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
                    "\tOutput axis is in the input axis, not need to interpolate, simple intersection")
                in1 = np.in1d(Caxis, self.axis)
                #                 C = C[:, in1]
                C = C[{vertical_dim: in1}]
            #                  da[dict(space=0)]

            elif (not self.issimilar(Caxis)):

                if self._debug: print("\tOutput axis is new, will use interpolation")
                [Np, Nz] = C.shape

                # Possibly Create a mixed layer for the interpolation to work
                # smoothly at the surface
                if ((Caxis[0] < 0.) & (self.axis[0] == 0.)):
                    # If data starts below the surface and feature axis requested is at the surface,
                    # we add one surface level to data, and mix it
                    Caxis = np.concatenate((np.zeros(1), Caxis))
                    x = np.empty((Np, 1))
                    x.fill(np.nan)
                    x = xr.DataArray(x, dims=['sampling', vertical_dim],
                                     coords={'sampling': C['sampling'], vertical_dim: np.zeros((1,))})
                    C = xr.concat((x, C), dim=vertical_dim)
                    # Fill in to the surface the 1st non-nan value (same as self.mix, but optmised):
                    C = C.bfill(dim=vertical_dim)

                    if self._debug:
                        print("\tData (%s) vertically mixed to reached the surface" % type(C))

                # Linear interpolation of profiles onto the model grid:
                # if self._debug: print("\tData type before interpolation: %s" % type(C))

                C = C.interp(coords={vertical_dim: self.axis})

                # if self._debug: print("\tData type after interpolation: %s" % type(C))

                # I don't understand why, but the interp2d return an array corresponding to a sorted Caxis.
                # Because our convention is to use a negative Caxis, oriented downward, i.e. not sorted, we
                # need to flip the interpolation results so that it's oriented along Caxis and not sort(Caxis).
            #                 if self.axis[0] > self.axis[-1]:
            #                     C = np.flip(C, axis=1)

            else:
                raise ValueError("I dont know how to vertically interpolate this array !")

        elif self._debug:
            print("\tThis is a slice array, no vertical interpolation")
        return C
