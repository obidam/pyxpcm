from pyxpcm.models import pcm
from pyxpcm.models import PCMFeatureError, PCMClassError
from sklearn.exceptions import NotFittedError
import pyxpcm
import numpy as np
import xarray as xr
import pytest
from utils import backends, backends_ids


@pytest.mark.parametrize("backend", backends, indirect=False, ids=backends_ids)
def test_pcm_init_req(backend):
    """Test PCM default instantiation with required arguments"""
    pcm_features_list = (
            {'F1': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': None})

    with pytest.raises(TypeError):
        m = pcm(backend=backend)

    with pytest.raises(TypeError):
        m = pcm(K=0, backend=backend)

    with pytest.raises(TypeError):
        m = pcm(features=pcm_features_list[0], backend=backend)

    with pytest.raises(PCMClassError):
        m = pcm(K=0, features=pcm_features_list[0], backend=backend)

    with pytest.raises(PCMFeatureError):
        m = pcm(K=1, features=dict(), backend=backend)

    for pcm_features in pcm_features_list:
        m = pcm(K=3, features=pcm_features, backend=backend)
        assert isinstance(m, pcm) == True

    with pytest.raises(NotFittedError):
        m = pcm(K=1, features=pcm_features_list[0], backend=backend)
        if not hasattr(m, 'fitted'):
            raise NotFittedError


@pytest.mark.parametrize("backend", backends, indirect=False, ids=backends_ids)
def test_pcm_init_opt(backend):
    """Test PCM instantiation with optional arguments"""
    pcm_features_list = (
            {'F1': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': np.arange(0, -500, 2.)},
            {'F1': np.arange(0, -500, 2.), 'F2': None})

    for pcm_features in pcm_features_list:

        for scaling in [0, 1, 2]:
            m = pcm(K=3, features=pcm_features, scaling=scaling, backend=backend)
            assert isinstance(m, pcm) == True

            for reduction in [0, 1]:
                m = pcm(K=3, features=pcm_features, scaling=scaling,
                        reduction=reduction, backend=backend)
                assert isinstance(m, pcm) == True

                m = pcm(K=3, features=pcm_features, scaling=scaling,
                        reduction=reduction, maxvar=90., backend=backend)
                assert isinstance(m, pcm) == True

                m = pcm(K=3, features=pcm_features, scaling=scaling,
                        reduction=reduction, maxvar=90.,
                        classif='gmm', backend=backend)
                assert isinstance(m, pcm) == True

                for covariance_type in ['full', 'diag', 'spherical']:
                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type, backend=backend)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, backend=backend)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, debug=0, backend=backend)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, debug=0,
                            timeit=True, backend=backend)
                    assert isinstance(m, pcm) == True

                    m = pcm(K=3, features=pcm_features, scaling=scaling,
                            reduction=reduction, maxvar=90.,
                            classif='gmm', covariance_type=covariance_type,
                            verb=False, debug=0,
                            timeit=True, timeit_verb=True, backend=backend)
                    assert isinstance(m, pcm) == True

