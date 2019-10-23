from pyxpcm.models import pcm
from pyxpcm.models import PCMFeatureError
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import pyxpcm
import numpy as np
import xarray as xr
import pytest

def new_m():
    return pcm(K=8, features={'temperature': np.arange(0, -500, 2.)})

def test_data_loader():
    """Test dummy dataset loader"""
    for d in ['dummy', 'argo', 'isas_snapshot', 'isas_series']:
        ds = pyxpcm.tutorial.open_dataset('argo').load()
        assert isinstance(ds, xr.Dataset) == True

def test_pcm_init():
    """ Test PCM instantiation

        Try to succeed in creating a PCM with all possible combination of option values
    """
    pcm_features_list = (
            {'F1': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': None})

    with pytest.raises(TypeError):
        m = pcm()

    with pytest.raises(TypeError):
        m = pcm(K=0)

    with pytest.raises(TypeError):
        m = pcm(features=pcm_features_list[0])

    with pytest.raises(PCMFeatureError):
        m = pcm(K=1, features=dict())

    for pcm_features in pcm_features_list:
        m = pcm(K=3, features=pcm_features)
        assert isinstance(m, pcm) == True

        for scaling in [0, 1, 2]:
            m = pcm(K=3, features=pcm_features, scaling=scaling)
            assert isinstance(m, pcm) == True

            for reduction in [0, 1]:
                m = pcm(K=3, features=pcm_features, scaling=scaling,
                        reduction=reduction)
                assert isinstance(m, pcm) == True

                m = pcm(K=3, features=pcm_features, scaling=scaling,
                        reduction=reduction, maxvar=90.)
                assert isinstance(m, pcm) == True

                m = pcm(K=3, features=pcm_features, scaling=scaling,
                        reduction=reduction, maxvar=90.,
                        classif='gmm')
                assert isinstance(m, pcm) == True

                for covariance_type in ['full', 'diag', 'spherical']:
                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, debug=True)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, debug=True,
                            timeit=True)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, debug=True,
                            timeit=True, timeit_verb=True)
                    assert isinstance(m, pcm) == True

    # Must not be trained yet:
    with pytest.raises(NotFittedError):
        assert check_is_fitted(m, 'fitted')