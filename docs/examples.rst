Example
=======

Create a PCM
------------
.. ipython:: python
   :suppress:

    import numpy as np

A Profile Classification Model can be created independently of any dataset using the :class:`pyxpcm.pcm` class constructor:

.. ipython:: python

    from pyxpcm import pcm
    m = pcm(K=8, feature_axis=np.arange(-500,0,2), feature_name='temperature')
    m

The PCM class constructor takes at least 3 arguments:

- ``K``: the number of class (integer) in the classification model,
- ``feature_axis``: the vertical axis (numpy array) that the PCM will work with (to train the classifier or to classify new data),
- ``feature_name``: the name (string) to be used when searching for the feature variable in an :class:`xarray.Dataset`.

In the above instantiation, we created a PCM with 8 classes that will work with *temperature* on a vertical axis with a
2m resolution from -500m to the surface. By default, the :class:`pyxpcm.pcm` class instance is set to use normalisation, dimensionality reduction with
Principal Component Analsysis and a Gaussian Mixture Model (GMM) as a classifier.

Load a dummy dataset
--------------------

To get you started, you can load a dummy sample dataset of Argo_ profiles interpolated on standard depth levels:

.. ipython:: python

    from pyxpcm import datasets as pcmdata
    ds = pcmdata.load_argo()
    ds

This is an :class:`xarray.Dataset` with 100 profiles with 21 depth levels. Each profile has physical variables such as temperature
(``TEMP``), salinity (``PSAL``), potentiel density (``SIG0``) and stratification (``BRV2``), along with geolocalisation information:
latitude (``LATITUDE``), longitude (``LONGITUDE``) and time (``TIME``).

To check out for a single profile, you can simply use the xarray plot method:

.. ipython:: python

    ds['TEMP'].isel(N_PROF=0).plot()
    @savefig examples_profile_sample.png width=5in

Fit the PCM
-----------

Now that we have a collection of profiles and a PCM, we can simply *fit* the classifier:

.. ipython:: python

    m.fit(ds, feature={'temperature': 'TEMP'})

where the :func:`pyxpcm.pcm.fit` method requires:

- a :class:`xarray.Dataset`, here ``ds``
- and a ``feature`` dictionnary-like argument with the ``feature_name`` argument used to instantiate the PCM as a key and
  with value, the :class:`xarray.DataArray` name holding this feature in the provided :class:`xarray.Dataset`.

In the above example we indicate to the PCM instance ``m`` that the feature named *temperature* is to be found in
``ds['TEMP']``.

By default the PCM uses a Gaussian Mixture Model as a classifier. It is computed using the scikit-learn :class:`sklearn.mixture.GaussianMixture`.


Classification and Prediction
-----------------------------

Once the PCM is trained, i.e. fitted with a training dataset, we can predict classes that profiles from a :class:`xarray.Dataset` belongs to. We can simply use the dummy dataset here:

.. ipython:: python

    LABELS = m.predict(ds, feature={'temperature': 'TEMP'})
    LABELS

And since by default the PCM classifier is fuzzy, we can also predict the probabilities for profiles to belong to each of the classes:

.. ipython:: python

    POSTERIORS = m.predict_proba(ds, feature={'temperature': 'TEMP'})
    POSTERIORS

.. _Argo: http://argo.ucsd.edu/
.. _Xarray: http://xarray.pydata.org/en/stable/data-structures.html#dataset
