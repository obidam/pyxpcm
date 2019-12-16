#!/bin/env python
# -*coding: UTF-8 -*-
#
# m = pyxpcm.load_netcdf(<path>)
# m.to_netcdf(<path>)
#
# Created by gmaze on 2019-10-15
__author__ = 'gmaze@ifremer.fr'

import os
import sys
import warnings
import numpy as np
import xarray as xr
import pickle
from datetime import datetime
import errno
from sklearn.utils import validation
from sklearn.mixture import GaussianMixture
from . import models
from . import __version__

# Define variables for netcdf load/save:
__software_name__ = 'Profile Classification Model - pyXpcm library'
__format_version__ = 2.0
# Version 1.0 was the version used by the Matlab library, limited to a single feature clustering.
# Instead of converting, we make sure to be able to load version 1.0

def _TransformerName(obj):
    return str(type(obj)).split('>')[0].split('.')[-1].split("'")[0]

def _load(file_path="pcm.nc"):
    """ Load a PCM file"""
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.nc':
        m = load_netcdf(file_path)
    else:
        with open(file_path, "rb") as f:
            m = pickle.load(f)
    return m

def _save(m, file_path="pcm.nc"):
    """ Save a PCM to file """
    filename, file_extension = os.path.splitext(file_path)
    if file_extension == '.nc':
        to_netcdf(m, file_path)
    else:
        with open(file_path, "wb") as f:
            pickle.dump(m, f)

def _load_netcdf_format2(ncfile):
    """ Load a PCM model from a netcdf file format 2.0

        Parameters
        ----------
        ncfile : str
            File name from which to load a PCM.


    """

    pcm2cdf = dict()
    pcm2cdf['global'] = xr.open_dataset(ncfile, group='/')

    if pcm2cdf['global'].attrs['software'] != __software_name__:
        raise ValueError("Can't loading netcdf not created with this software.\n" +
                      pcm2cdf['global'].attrs['software'])
    if pcm2cdf['global'].attrs['format_version'] != __format_version__:
        raise ValueError("Incompatible format version " + pcm2cdf['global'].attrs['format_version'])

    for feature in pcm2cdf['global']['feature'].values:
        pcm2cdf[feature] = xr.open_dataset(ncfile, group=("feature_%s" % feature))
    pcm2cdf['classifier'] = xr.open_dataset(ncfile, group='classifier')

    # Create a new pcm instance:
    K = pcm2cdf['global']['K'].shape[0]
    scal = {'none': 0, 'normal': 1, 'center': 2}
    scaling = scal[pcm2cdf['global'].attrs['scaler']]
    reduction = eval(str(pcm2cdf['global'].attrs['reducer']))
    maxvar = int(pcm2cdf['global'].attrs['reducer_maxvar'])
    classif = pcm2cdf['classifier'].attrs['type']
    covariance_type = pcm2cdf['classifier'].attrs['covariance_type']
    backend = pcm2cdf['global'].attrs['backend']

    features = dict()
    for feature in pcm2cdf['global']['feature'].values:
        features[feature] = pcm2cdf[feature]['Z'].values

    loaded_m = models.pcm(K,
                   features=features,
                   scaling=scaling,
                   reduction=reduction, maxvar=maxvar,
                   classif=classif, covariance_type=covariance_type,
                   backend=backend)

    # Fill new instance with fitted method information:
    for feature in loaded_m.features:

        if eval(pcm2cdf['global'].attrs['fitted']):
            # Scaler:
            if pcm2cdf['global'].attrs['scaler'] in ['normal', 'center']:
                loaded_m._scaler[feature].mean_ = pcm2cdf[feature]['scaler_center'].values
                if pcm2cdf['global'].attrs['scaler'] == 'normal':
                    loaded_m._scaler[feature].scale_ = pcm2cdf[feature]['scaler_scale'].values
                else:
                    setattr(loaded_m._scaler[feature], 'scale_', None)

                setattr(loaded_m._scaler[feature], 'fitted', True)
                validation.check_is_fitted(loaded_m._scaler[feature], 'fitted')

            # Reducer:
            if reduction:
                loaded_m._reducer[feature].mean_ = pcm2cdf[feature]['reducer_center'].values
                loaded_m._reducer[feature].components_ = pcm2cdf[feature]['reducer_eigenvector'].values
                setattr(loaded_m._reducer[feature], 'fitted', True)
                validation.check_is_fitted(loaded_m._reducer[feature], 'fitted')

            # Homogeniser:
            loaded_m._homogeniser[feature]['mean'] = pcm2cdf[feature].attrs['homogeniser'][0]
            loaded_m._homogeniser[feature]['std'] = pcm2cdf[feature].attrs['homogeniser'][1]

    # Classifier:
    if eval(pcm2cdf['global'].attrs['fitted']):
        gmm = GaussianMixture(n_components=loaded_m.K,
                              covariance_type=covariance_type,
                              n_init=0, max_iter=1, warm_start=True,
                              weights_init=pcm2cdf['classifier']['prior'].values,
                              means_init=pcm2cdf['classifier']['center'].values,
                              precisions_init=pcm2cdf['classifier']['precision'].values)
        setattr(gmm, 'fitted', True)
        setattr(gmm, 'weights_', pcm2cdf['classifier']['prior'].values)
        setattr(gmm, 'means_', pcm2cdf['classifier']['center'].values)
        setattr(gmm, 'precisions_cholesky_', pcm2cdf['classifier']['precision_cholesky'].values)
        loaded_m._classifier = gmm
        validation.check_is_fitted(gmm, 'fitted')

    # PCM properties
    if eval(pcm2cdf['global'].attrs['fitted']):
        loaded_m._props['llh'] = pcm2cdf['global'].attrs['fit_score']
        setattr(loaded_m, 'fitted', True)

    return loaded_m

def _load_netcdf_format1(ncfile):
    """ Load a PCM model from a netcdf file with format version 1.0

        Model loader for Matlab PCM files

        Parameters
        ----------
        ncfile : str
            File name from which to load a PCM.

    """

    pcm2cdf = dict()
    pcm2cdf['global'] = xr.open_dataset(ncfile, group='/')

    if pcm2cdf['global'].attrs['software'] != "Profile Classification Model - Matlab Toolbox (c) Ifremer":
        raise ValueError("Can't load netcdf not created with appropriate software.\n" +
                         pcm2cdf['global'].attrs['software'])
    if pcm2cdf['global'].attrs['format_version'] != "1.0":
        raise ValueError("Incompatible format version " + pcm2cdf['global'].attrs['format_version'])

    pcm2cdf['scaler'] = xr.open_dataset(ncfile, group='Normalization')
    pcm2cdf['reducer'] = xr.open_dataset(ncfile, group='Reduction')
    pcm2cdf['classifier'] = xr.open_dataset(ncfile, group='ClassificationModel')

    # Create a new pcm instance:
    K = len(pcm2cdf['classifier']['CLASS'])

    scaling = int(pcm2cdf['global'].attrs['PCM_normalization'])

    reduction = False
    if pcm2cdf['global'].attrs['PCM_doREDUCE'] == 'true':
        reduction = True
    #     maxvar = int(pcm2cdf['global'].attrs['reducer_maxvar'])
    maxvar = len(pcm2cdf['reducer']['REDUCED_DIM'])

    classif = 'gmm'
    covariance_type = pcm2cdf['classifier'].attrs['Covariance_matrix_original_form']
    backend = 'sklearn'

    feature = 'unknown'
    features = {feature: pcm2cdf['global']['DEPTH_MODEL']}

    loaded_m = pyxpcm.models.pcm(K,
                                 features=features,
                                 scaling=scaling,
                                 reduction=reduction, maxvar=maxvar,
                                 classif=classif, covariance_type=covariance_type,
                                 backend=backend)

    # Fill new instance with fitted method information:

    # Scaler:
    if scaling in [1, 2]:
        loaded_m._scaler[feature].mean_ = pcm2cdf['scaler']['X_ave'].values
        if scaling == 1:
            loaded_m._scaler[feature].scale_ = pcm2cdf['scaler']['X_std'].values
        else:
            setattr(loaded_m._scaler[feature], 'scale_', None)
        setattr(loaded_m._scaler[feature], 'fitted', True)
        validation.check_is_fitted(loaded_m._scaler[feature], 'fitted')

    # Reducer:
    if reduction:
        loaded_m._reducer[feature].mean_ = pcm2cdf['reducer']['X_ref'].values
        loaded_m._reducer[feature].components_ = pcm2cdf['reducer']['EOFs'].values
        setattr(loaded_m._reducer[feature], 'fitted', True)
        validation.check_is_fitted(loaded_m._reducer[feature], 'fitted')

    # Classifier:
    gmm = GaussianMixture(n_components=loaded_m.K,
                          covariance_type=covariance_type,
                          n_init=0, max_iter=1, warm_start=True,
                          weights_init=pcm2cdf['classifier']['priors'].values,
                          means_init=pcm2cdf['classifier']['centers'].values,
                          precisions_init=np.linalg.inv(pcm2cdf['classifier']['covariances'].values))
    setattr(gmm, 'fitted', True)
    setattr(gmm, 'weights_', pcm2cdf['classifier']['priors'].values)
    setattr(gmm, 'means_', pcm2cdf['classifier']['centers'].values)
    setattr(gmm, 'precisions_cholesky_', np.linalg.cholesky(np.linalg.inv(pcm2cdf['classifier']['covariances'].values)))
    loaded_m._classifier = gmm
    validation.check_is_fitted(gmm, 'fitted')

    # PCM properties
    loaded_m._props['llh'] = pcm2cdf['classifier']['llh'].values
    setattr(loaded_m, 'fitted', True)

    return loaded_m

def to_netcdf(m, ncfile=None, global_attributes=dict(), mode='w'):
    """ Save a PCM to a netcdf file

        Any existing file at this location will be overwritten by default.
        Time logging information are not saved.

        Parameters
        ----------
        ncfile : str
            File name where to save this PCM.

        global_attributes: dict()
            Dictionnary of attributes to add to the Netcdf4 file under the global scope.

        mode : str
            Writing mode of the file.
            mode='w' (default) overwrite any existing file.
            Anything else will raise an Error if file exists.
    """

    if (mode == 'w' and os.path.exists(ncfile) and os.path.isfile(ncfile)):
        os.remove(ncfile)
    elif (os.path.exists(ncfile) and os.path.isfile(ncfile)):
        raise OSError(errno.EEXIST,
                      "File exists. Use mode='w' to overwrite.  ",
                      ncfile)

    # Check if we know how to save predictors:
    for feature in m.features:
        scalers_ok = ['StandardScaler',
                      'NoTransform']
        if _TransformerName(m._scaler[feature]) not in scalers_ok:
            raise TypeError("Export to netcdf is not supported for scaler of type: " +
                             str(type(m._scaler[feature])))
        reducers_ok = ['PCA',
                       'NoTransform']
        if _TransformerName(m._reducer[feature]) not in reducers_ok:
            raise TypeError("Export to netcdf is not supported for reducer of type: " +
                             str(type(m._reducer[feature])))
    if not isinstance(m._classifier, GaussianMixture):
        raise TypeError("Export to netcdf is not supported for classifier of type: " +
                         str(type(m._classifier)))

    # Create the list of xr.dataset that should go into different scope in the netcdf 4 file
    pcm2cdf = dict()

    # Create global scope
    feature_names = [feature for feature in m.features]
    ds_global = xr.merge([
        xr.DataArray(feature_names, name='feature', dims='F', coords={'F': np.arange(0, m.F)}),
        xr.DataArray(np.arange(0, m.K), name='class', dims='K', coords={'K': np.arange(0, m.K)}),
    ])
    ds_global.attrs['backend'] = m.backend
    if hasattr(m, 'fitted'):
        ds_global.attrs['fitted'] = str(m.fitted)
        ds_global.attrs['fit_datetime'] = str(m.fitstats['datetime'])
        ds_global.attrs['fit_score'] = m.fitstats['score']
        ds_global.attrs['fit_n_samples_seen'] = m.fitstats['n_samples_seen_']
    else:
        ds_global.attrs['fitted'] = str(False)
    ds_global.attrs['scaler'] = m._props['with_scaler']
    ds_global.attrs['reducer'] = str(m._props['with_reducer'])
    ds_global.attrs['reducer_maxvar'] = str(m._props['maxvar'])

    ds_global.attrs['software'] = __software_name__
    ds_global.attrs['software_version'] = __version__
    ds_global.attrs['format_version'] = __format_version__
    # Add user defined additional global attributes:
    for key in global_attributes:
        ds_global.attrs[key] = global_attributes[key]
    pcm2cdf['global'] = ds_global

    # Create feature scopes:
    for feature in m.features:
        if hasattr(m, 'fitted'):
            if m._props['with_scaler'] == 'normal' or m._props['with_scaler'] == 'center':
                ds_scaler = xr.DataArray(m._scaler[feature].mean_, name='scaler_center',
                                         dims='Z', coords={'Z': m._props['features'][feature]},
                                         attrs={'feature': feature,
                                                'long_name': 'scaler mean',
                                                'unit': m._scaler_props[feature]['units']}).to_dataset()
            if m._props['with_scaler'] == 'normal':
                ds_scaler = xr.merge([ds_scaler,
                                      xr.DataArray(m._scaler[feature].scale_, name='scaler_scale',
                                                   dims='Z', coords={'Z': m._props['features'][feature]},
                                                   attrs={'feature': feature,
                                                          'long_name': 'scaler std',
                                                          'unit': m._scaler_props[feature]['units']})
                                      ])
            if m._props['with_reducer']:
                Xeof_mean = m._reducer[feature].mean_
                Xeof = m._reducer[feature].components_
                ds_reducer = xr.merge([
                    xr.DataArray(Xeof_mean, name='reducer_center',
                                 dims=['Z'],
                                 coords={'Z': m._props['features'][feature]},
                                 attrs={'feature': feature,
                                        'long_name': 'PCA center'}),
                    xr.DataArray(Xeof, name='reducer_eigenvector',
                                 dims=['REDUCED_DIM', 'Z'],
                                 coords={'Z': m._props['features'][feature],
                                         'REDUCED_DIM': np.arange(0, Xeof.shape[0])},
                                 attrs={'feature': feature,
                                        'long_name': 'PCA eigen vectors',
                                        'maxvar': str(m._props['maxvar'])})
                ])

            if 'none' not in m._props['with_scaler'] and m._props['with_reducer']:
                ds_feature = xr.merge([ds_scaler, ds_reducer])
            elif 'none' in m._props['with_scaler'] and m._props['with_reducer']:
                ds_feature = ds_reducer
            elif 'none' not in m._props['with_scaler'] and not m._props['with_reducer']:
                ds_feature = ds_scaler
            else:
                ds_feature = xr.DataArray(m._props['features'][feature], name='Z',
                                          dims='N_LEVELS',
                                          attrs={'feature': feature,
                                                 'long_name': 'Vertical_axis'}).to_dataset()
            ds_feature.attrs['homogeniser'] = np.array(
                [m._homogeniser[feature]['mean'], m._homogeniser[feature]['std']])
        else:
            ds_feature = xr.DataArray(m._props['features'][feature], name='Z',
                                      dims='N_LEVELS',
                                      attrs={'feature': feature,
                                             'long_name': 'Vertical_axis'}).to_dataset()
        ds_feature.attrs['name'] = feature
        pcm2cdf[feature] = ds_feature

    # Create classifier scope:
    if hasattr(m, 'fitted'):

        if m._props['with_reducer']:
            GMM_DIM = np.arange(0, np.sum([len(pcm2cdf[f]['REDUCED_DIM']) for f in m.features]))
        else:
            GMM_DIM = np.arange(0, np.sum([len(pcm2cdf[f]['Z']) for f in m.features]))

        if m._classifier.covariance_type == 'full':
            covars = m._classifier.covariances_
            precisions = m._classifier.precisions_
            precisions_cholesky = m._classifier.precisions_cholesky_

        elif m._classifier.covariance_type == 'diag':
            covars = np.zeros((m.K, len(GMM_DIM), len(GMM_DIM)))
            precisions = np.zeros((m.K, len(GMM_DIM), len(GMM_DIM)))
            precisions_cholesky = np.zeros((m.K, len(GMM_DIM), len(GMM_DIM)))
            for ik, k in enumerate(m):
                covars[k, :, :] = np.diag(m._classifier.covariances_[ik, :])
                precisions[k, :, :] = np.diag(m._classifier.precisions_[ik, :])
                precisions_cholesky[k, :, :] = np.diag(m._classifier.precisions_cholesky_[ik, :])

        elif m._classifier.covariance_type == 'spherical':
            covars = np.zeros((m.K, len(GMM_DIM), len(GMM_DIM)))
            precisions = np.zeros((m.K, len(GMM_DIM), len(GMM_DIM)))
            precisions_cholesky = np.zeros((m.K, len(GMM_DIM), len(GMM_DIM)))
            for ik, k in enumerate(m):
                covars[k, :, :] = m._classifier.covariances_[ik] * np.eye((len(GMM_DIM)))
                precisions[k, :, :] = m._classifier.precisions_[ik] * np.eye((len(GMM_DIM)))
                precisions_cholesky[k, :, :] = m._classifier.precisions_cholesky_[ik] * np.eye((len(GMM_DIM)))

        ds_classifier = xr.merge([
            xr.DataArray(m._classifier.weights_, name='prior',
                         dims='K',
                         coords={'K': pcm2cdf['global']['K']},
                         attrs={'long_name': 'Mixture component priors'}),
            xr.DataArray(m._classifier.means_, name='center',
                         dims=['K', 'GMM_DIM'],
                         coords={'K': pcm2cdf['global']['K'],
                                 'GMM_DIM': GMM_DIM},
                         attrs={'long_name': 'Mixture component centers'}),
            xr.DataArray(covars, name='covariance',
                         dims=['K', 'GMM_DIM', 'GMM_DIM'],
                         attrs={'long_name': 'Mixture component covariances',
                                'initial_shape': m._props['COVARTYPE']}),
            xr.DataArray(precisions, name='precision',
                         dims=['K', 'GMM_DIM', 'GMM_DIM'],
                         attrs={'long_name': 'Mixture component precisions',
                                'comment': 'A precision matrix is the inverse of a covariance matrix.' + \
                                           'Storing the precision matrices instead of the covariance matrices makes ' + \
                                           'it more efficient to compute the log-likelihood of new samples.',
                                'initial_shape': m._props['COVARTYPE']}),
            xr.DataArray(precisions_cholesky, name='precision_cholesky',
                         dims=['K', 'GMM_DIM', 'GMM_DIM'],
                         attrs={'long_name': 'Mixture component Cholesky precisions',
                                'comment': 'Cholesky decomposition of the precision matrices of each mixture component.',
                                'initial_shape': m._props['COVARTYPE']}),
            xr.DataArray(m._xlabel, name='dimension_name',
                         dims=['GMM_DIM'],
                         attrs={'long_name': 'Name of reduced dimensions'})
        ])
        ds_classifier.attrs['type'] = m._props['with_classifier']
        ds_classifier.attrs['covariance_type'] = m._props['COVARTYPE']
    else:
        ds_classifier = xr.Dataset()
        ds_classifier.attrs['type'] = m._props['with_classifier']
        ds_classifier.attrs['covariance_type'] = m._props['COVARTYPE']
    pcm2cdf['classifier'] = ds_classifier

    # Create file
    pcm2cdf['global'].attrs['creation_date'] = str(datetime.utcnow())
    pcm2cdf['global'].to_netcdf(ncfile, mode='w', format='NETCDF4', group='/')
    for feature in m.features:
        pcm2cdf[feature].to_netcdf(ncfile, mode='a', format='NETCDF4', group='feature_' + feature)
    #     if hasattr(m, 'fitted'):
    pcm2cdf['classifier'].to_netcdf(ncfile, mode='a', format='NETCDF4', group='classifier')

def load_netcdf(ncfile):
    """ Load a PCM model from netcdf file

        Parameters
        ----------
        ncfile : str
            File name from which to load a PCM.


    """

    pcm2cdf = dict()
    pcm2cdf['global'] = xr.open_dataset(ncfile, group='/')

    # Check file format:
    if pcm2cdf['global'].attrs['format_version'] == "1.0":
        loaded_m = _load_netcdf_format1(ncfile)
    elif pcm2cdf['global'].attrs['format_version'] != __format_version__:
        raise ValueError("Incompatible format version " + pcm2cdf['global'].attrs['format_version'])
    else:
        loaded_m = _load_netcdf_format2(ncfile)
    return loaded_m
