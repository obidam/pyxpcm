from pyxpcm.models import pcm
from pyxpcm.models import PCMFeatureError, PCMClassError
from sklearn.exceptions import NotFittedError
import pyxpcm
import numpy as np
import xarray as xr
import pytest

def test_pcm_init_req():
    """Test PCM default instantiation with required arguments"""
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

    with pytest.raises(PCMClassError):
        m = pcm(K=0, features=pcm_features_list[0])

    with pytest.raises(PCMFeatureError):
        m = pcm(K=1, features=dict())

    for pcm_features in pcm_features_list:
        m = pcm(K=3, features=pcm_features)
        assert isinstance(m, pcm) == True

    with pytest.raises(NotFittedError):
        m = pcm(K=1, features=pcm_features_list[0])
        if not hasattr(m, 'fitted'):
            raise NotFittedError

def test_pcm_init_opt():
    """Test PCM instantiation with optional arguments"""
    pcm_features_list = (
            {'F1': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': None})

    for pcm_features in pcm_features_list:

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

