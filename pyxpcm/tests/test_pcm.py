from pyxpcm.pcmodel import pcm
from pyxpcm.pcmodel import PCMFeatureError
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from pyxpcm import datasets as pcmdata
import numpy as np
import xarray as xr
import pytest

def new_m():
    return pcm(K=8, feature_axis=np.arange(-500, 0, 2), feature_name='temperature')

def test_data_loader():
    """Test dummy dataset loader"""
    ds = pcmdata.load_argo()
    assert isinstance(ds, xr.Dataset) == True
    assert 'DEPTH' in ds.dims
    assert 'N_PROF' in ds.dims
    assert 'TEMP' in ds.data_vars

def test_pcm_init():
    """Test the different PCM instantiation methods
        This is not where we validate arguments.
    """

    m = pcm()
    assert isinstance(m,pcm) == True

    m = pcm(K=12)
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2))
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature')
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature', scaling=1)
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature', scaling=1,
            reduction=True)
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature', scaling=1,
            reduction=True, maxvar=90.)
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature', scaling=1,
            reduction=True, maxvar=90., classif='gmm')
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature', scaling=1,
            reduction=True, maxvar=90., classif='gmm', covariance_type='full')
    assert isinstance(m,pcm) == True

    m = pcm(K=12, feature_axis=np.arange(-500, 0, 2), feature_name='temperature', scaling=1,
            reduction=True, maxvar=90., classif='gmm', covariance_type='full', verb=False)
    assert isinstance(m,pcm) == True

    # Must not bet trained yet:
    with pytest.raises(NotFittedError):
        assert check_is_fitted(m, 'fitted')

    with pytest.raises(NotFittedError):
        check_is_fitted(m._classifier, 'weights_')

def test_pcm_preprocessing():
    """Test PCM Data preprocessing method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Create model:
    m = new_m()

    # Pre-processing:

    with pytest.raises(PCMFeatureError):
        X = m.preprocessing(ds)
        # This must raise an error because 'temperature' is not in 'ds'

    with pytest.raises(PCMFeatureError):
        X = m.preprocessing(ds, feature={'john':'doe'})
        # This must raise an error because 'john' is not 'temperature'

    with pytest.raises(PCMFeatureError):
        X = m.preprocessing(ds, feature={'temperature':'john'})
        # This must raise an error because 'john' is not a variable in 'ds'

    X = m.preprocessing(ds, feature={'temperature': 'TEMP'}) # This must work
    assert len(X.shape) == 2, "Working array must be 2-dimensional"

    # Training
    # m.fit(ds, feature={'temperature': 'TEMP'})

    # Classifying
    # m.predict(ds, feature={'temperature': 'TEMP'}, inplace=True)
    # m.predict_proba(ds, feature={'temperature': 'TEMP'}, inplace=True)

def test_pcm_fit():
    """Test PCM Data fit method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Create model:
    m = new_m()

    # Successful training:
    m.fit(ds, feature={'temperature': 'TEMP'})
    assert m.fitted == True, "'fitted' property must be true after training"
    assert check_is_fitted(m, 'fitted') is None
    assert check_is_fitted(m._classifier, 'weights_') is None

    # Now assume the feature name is the real variable name in the dataset:
    ds = pcmdata.load_argo()
    ds['temperature'] = ds['TEMP']
    ds = ds.drop('TEMP')
    m = new_m() # feature is set to 'temperature' by default in tests
    m.fit(ds)
    assert m.fitted == True, "'fitted' property must be true after training"
    assert check_is_fitted(m, 'fitted') is None
    assert check_is_fitted(m._classifier, 'weights_') is None

    # Training leading to errors because feature data cannot be found:
    with pytest.raises(PCMFeatureError):
        m.fit(ds, feature={'temperature': 'TEMP'})

    with pytest.raises(PCMFeatureError):
        m.fit(ds, feature='john')

    with pytest.raises(PCMFeatureError):
        m.fit(ds, feature={'doe': 'john'})

    with pytest.raises(PCMFeatureError):
        m.fit(ds, feature={'temperature': 'john'})

def test_pcm_predict():
    """Test PCM Data predict method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Create model:
    m = new_m()

    # Training:
    m.fit(ds, feature={'temperature': 'TEMP'})

    # Classifying

    # Successful predictions:
    labels = m.predict(ds, feature={'temperature': 'TEMP'})
    assert isinstance(labels, xr.DataArray) == True, "Output must be a xarray.DataArray"
    assert np.all(np.isin(np.unique(labels.values), np.arange(0,m.K))) == True, "Output content must be integers between 0 and K"

    labels = m.predict(ds, feature={'temperature': 'TEMP'}, name='CUSTOM_NAME')
    assert labels.name == 'CUSTOM_NAME', "Output xarray.DataArray must be named according to argument 'name'"

    dsl = ds.copy()
    m.predict(dsl, feature={'temperature': 'TEMP'}, inplace=True)
    assert isinstance(dsl, xr.Dataset) == True, "Input xarray.DataSet must remain a xarray.DataSet"
    labelname = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    labels = dsl[labelname]
    assert isinstance(labels, xr.DataArray) == True, "Output must be a xarray.DataArray"
    assert np.all(np.isin(np.unique(labels.values), np.arange(0,m.K))) == True, "Output content must be integers between 0 and K"

    dsl = ds.copy()
    m.predict(dsl, feature={'temperature': 'TEMP'}, inplace=True, name='CUSTOM_NAME')
    assert isinstance(dsl, xr.Dataset) == True, "Input xarray.DataSet must remain a xarray.DataSet"
    labelname = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    assert labelname == 'CUSTOM_NAME', "New xarray.DataSet DataArray must be named according to argument 'name'"
    labels = dsl[labelname]
    assert isinstance(labels, xr.DataArray) == True, "Output must be a xarray.DataArray"
    assert np.all(np.isin(np.unique(labels.values), np.arange(0,m.K))) == True, "Output content must be integers between 0 and K"

    # Error in predictions:
    # Because not trained:
    m = new_m()
    with pytest.raises(NotFittedError):
        m.predict(ds, feature={'temperature': 'TEMP'})

def test_pcm_fit_predict():
    """Test PCM Data fit_predict method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Successful predictions:
    m = new_m()
    labels = m.fit_predict(ds, feature={'temperature': 'TEMP'})

    # Successful training:
    assert m.fitted == True, "'fitted' property must be true after training"
    assert check_is_fitted(m, 'fitted') is None
    assert check_is_fitted(m._classifier, 'weights_') is None

    # Successful output:
    assert isinstance(labels, xr.DataArray) == True, "Output must be a xarray.DataArray"
    assert np.all(np.isin(np.unique(labels.values), np.arange(0,m.K))) == True, "Output content must be integers between 0 and K"

    m = new_m()
    labels = m.fit_predict(ds, feature={'temperature': 'TEMP'}, name='CUSTOM_NAME')
    assert labels.name == 'CUSTOM_NAME', "Output xarray.DataArray must be named according to argument 'name'"

    dsl = ds.copy()
    m = new_m()
    m.fit_predict(dsl, feature={'temperature': 'TEMP'}, inplace=True)
    assert isinstance(dsl, xr.Dataset) == True, "Input xarray.DataSet must remain a xarray.DataSet"
    labelname = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    labels = dsl[labelname]
    assert isinstance(labels, xr.DataArray) == True, "Output must be a xarray.DataArray"
    assert np.all(np.isin(np.unique(labels.values), np.arange(0,m.K))) == True, "Output content must be integers between 0 and K"

    dsl = ds.copy()
    m = new_m()
    m.fit_predict(dsl, feature={'temperature': 'TEMP'}, inplace=True, name='CUSTOM_NAME')
    assert isinstance(dsl, xr.Dataset) == True, "Input xarray.DataSet must remain a xarray.DataSet"
    labelname = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    assert labelname == 'CUSTOM_NAME', "New xarray.DataSet DataArray must be named according to argument 'name'"
    labels = dsl[labelname]
    assert isinstance(labels, xr.DataArray) == True, "Output must be a xarray.DataArray"
    assert np.all(np.isin(np.unique(labels.values), np.arange(0,m.K))) == True, "Output content must be integers between 0 and K"

def test_pcm_predict_proba():
    """Test PCM Data predict_proba method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Training:
    m = new_m()
    m.fit(ds, feature={'temperature': 'TEMP'})

    # Classifying
    # def predict_proba(self, ds, feature=None, inplace=False, name='PCM_POST', classdimname='N_CLASS'):

    def assert_output_value_type(post):
        assert isinstance(post, xr.DataArray) == True, "Output must be a xarray.DataArray"

    def assert_output_value_range(post):
        assert np.all(np.bitwise_and(post >= 0, post <= 1)) == True, "Output content must be floats between 0 and 1"

    def assert_inoutput_value_type(ds):
        assert isinstance(ds, xr.Dataset) == True, "Input xarray.DataSet must remain a xarray.DataSet"

    # Successful predictions:
    post = m.predict_proba(ds, feature={'temperature': 'TEMP'})
    assert_output_value_type(post)
    assert_output_value_range(post)

    post = m.predict_proba(ds, feature={'temperature': 'TEMP'}, name='CUSTOM_NAME')
    assert post.name == 'CUSTOM_NAME', "Output xarray.DataArray must be named according to argument 'name'"

    post = m.predict_proba(ds, feature={'temperature': 'TEMP'}, name='CUSTOM_NAME', classdimname='CUSTOM_DIMNAME')
    assert post.name == 'CUSTOM_NAME', "Output xarray.DataArray must be named according to argument 'name'"
    assert post.dims[1] == 'CUSTOM_DIMNAME', "Output xarray.DataArray new classifier dimension must be named according to argument 'classdimname'"

    dsl = ds.copy()
    m.predict_proba(dsl, feature={'temperature': 'TEMP'}, inplace=True)
    assert_inoutput_value_type(dsl)
    name = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    post = dsl[name]
    assert_output_value_type(post)
    assert_output_value_range(post)

    dsl = ds.copy()
    m.predict_proba(dsl, feature={'temperature': 'TEMP'}, inplace=True, name='CUSTOM_NAME')
    assert_inoutput_value_type(dsl)
    name = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    assert name == 'CUSTOM_NAME', "New xarray.DataSet DataArray must be named according to argument 'name'"
    post = dsl[name]
    assert_output_value_type(post)
    assert_output_value_range(post)

    dsl = ds.copy()
    m.predict_proba(dsl, feature={'temperature': 'TEMP'}, inplace=True, name='CUSTOM_NAME', classdimname='CUSTOM_DIMNAME')
    assert_inoutput_value_type(dsl)
    name = list(set(dsl.data_vars)-set(ds.data_vars))[0] # Name of the new variable in the dataset
    assert name == 'CUSTOM_NAME', "New xarray.DataSet DataArray variable must be named according to argument 'name'"
    post = dsl[name]
    assert_output_value_type(post)
    assert_output_value_range(post)
    assert post.dims[1] == 'CUSTOM_DIMNAME', "New xarray.DataSet DataArray variable new classifier dimension must be named according to argument 'classdimname'"

    # Error in predictions:
    # Because not trained:
    m = new_m()
    with pytest.raises(NotFittedError):
        m.predict_proba(ds, feature={'temperature': 'TEMP'})

def test_pcm_score():
    """Test PCM Data score method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Training:
    m = new_m()
    m.fit(ds, feature={'temperature': 'TEMP'})

    #
    llh = m.score(ds, feature={'temperature': 'TEMP'})
    assert isinstance(llh, np.float64) == True, "Output score must a np.float64"

def test_pcm_bic():
    """Test PCM Data bic method"""
    # Load dummy data to work with:
    ds = pcmdata.load_argo()

    # Training:
    m = new_m()
    m.fit(ds, feature={'temperature': 'TEMP'})

    #
    bic = m.bic(ds, feature={'temperature': 'TEMP'})
    assert isinstance(bic, np.float64) == True, "Output BIC must a np.float64"

