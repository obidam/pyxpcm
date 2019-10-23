Basic example
=============

Here is a quick example of what you can do and how to use :class:`pyxpcm.pcm`.

To begin with, import PCM and numpy:

.. ipython:: python

    import numpy as np
    from pyxpcm.models import pcm

Create a PCM
------------

A Profile Classification Model (PCM) can be created independently of any dataset using the :class:`pyxpcm.pcm` class constructor:

.. ipython:: python

    m = pcm(K=3, features={'temperature': np.arange(0,-700.,-10.)})
    m

The PCM class constructor takes at least 2 arguments:

- ``K``: the number of class (integer) in the classification model,
- ``features``: a dictionnary where keys are feature name and values the corresponding vertical axis (numpy array) that the PCM will work with.

In the above instantiation, we created a PCM with 3 classes that will work with *temperature* on a vertical axis with a 10m resolution from -700m to the surface. By default, the :class:`pyxpcm.pcm` class instance is set to use vertical interpolation if necessary. Other class parameters allows to set normalisation, dimensionality reduction and classifier parameters.

Load a dataset
--------------

To get you started, you can load a sample dataset with Argo_ profiles interpolated on standard depth levels.

.. ipython:: python
    :okwarning:

    from pyxpcm import datasets as pcmdata
    ds = pcmdata.load_argo()
    ds

This is an :class:`xarray.Dataset` with 7560 profiles with 282 depth levels between the surface and -1405m. Each profile has physical variables such as temperature
(``TEMP``), salinity (``PSAL``), potentiel density (``SIG0``) and stratification (``BRV2``), along with geolocalisation information:
latitude (``LATITUDE``), longitude (``LONGITUDE``) and time (``TIME``). Note that this is a collection of profiles, the collection dimension being ``N_PROF``.

To visualise a single profile, you can simply use the Xarray_ plot method:

.. ipython:: python

    @savefig examples_profile_sample.png width=5in height=6in
    ds['TEMP'].isel(N_PROF=0).plot(y='DEPTH')


Fit the PCM on the dataset
--------------------------

Now that we have a collection of profiles and a PCM, we can simply use a :class:`xarray.Dataset` to *fit* the classifier parameters:

.. ipython:: python

    ds.pyxpcm.fit(m, features={'temperature': 'TEMP'}, dim='DEPTH')
    # or equivalently:
    m.fit(ds, features={'temperature': 'TEMP'})

where the :func:`pyxpcm.pcm.fit` method requires:

- a :class:`xarray.Dataset`, here ``ds``
- and a ``feature`` dictionnary-like argument with the ``feature_name`` argument used to instantiate the PCM as a key and
  with value, the :class:`xarray.DataArray` name holding this feature in the provided :class:`xarray.Dataset`.

In the above example we indicate to the PCM instance ``m`` that the feature named *temperature* is to be found in
``ds['TEMP']``.

Note that, at this time, pyXpcm assumes a scikit-learn convention whereby the first dimension of the 2-dimensional array from :class:`xarray.DataArray` is the sampling dimension (profiles) and the second dimension is the vertical depth axis.

By default the PCM uses a Gaussian Mixture Model as a classifier. It is computed using the scikit-learn :class:`sklearn.mixture.GaussianMixture`. In the future, other classifiers will be implemented `(see this issue) <https://github.com/obidam/pyxpcm/issues/5>`_.

Classify ocean profiles
-----------------------

There are two methods to classify ocean profiles:

Hard labelling
^^^^^^^^^^^^^^

Once the PCM is trained, i.e. fitted with a training dataset, we can predict classes that profiles from a :class:`xarray.Dataset` belongs to. We can simply classify profile from the dummy dataset that was used to fit the PCM:

.. ipython:: python

    LABELS = ds.pyxpcm.predict(m, features={'temperature': 'TEMP'})
    LABELS

Each profiles is labelled with one of the possible cluster index from 0 to K-1. Note that prediction can be ran on another collection of profiles, as long as they have temperature.

Fuzzy classification
^^^^^^^^^^^^^^^^^^^^

Since the PCM classifier we used (GMM) is fuzzy, we can also predict the probabilities for profiles to belong to each of the classes, the so-called posterior probabilities:

.. ipython:: python

    POSTERIORS = ds.pyxpcm.predict_proba(m, features={'temperature': 'TEMP'})
    POSTERIORS

In this case, a new dimension appears: ``pcm_class``. The sum over `pcm_class`` of the posterior probabilities is necessarily 1. We'll note that ``LABELS`` are the ``pcm_class`` index for which the posterior is maximum.

Add PCM results to the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Note that since we're working with Xarray_, one can add these new variables directly to the dataset as new variables. This is done using the ``inplace`` option:

.. ipython:: python

    ds = ds.pyxpcm.predict(m, features={'temperature': 'TEMP'}, inplace=True)
    ds = ds.pyxpcm.predict_proba(m, features={'temperature': 'TEMP'}, inplace=True)
    ds

We see that the ``ds`` object has two new variables added by each of these methods, the ``PCM_LABELS`` and ``PCM_POST``.
The new variable name can be tuned to your convenience using the ``name`` option. See more details in the :doc:`API reference </api>` (:func:`pyxpcm.pcm.predict` and :func:`pyxpcm.pcm.predict_proba`).

Summary
-------

You can look at in the :doc:`/summary` page for an more an overview.

.. _Argo: http://argo.ucsd.edu/
.. _Xarray: http://xarray.pydata.org/en/stable/
