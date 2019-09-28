.. currentmodule:: pyxpcm

#############
API reference
#############

This page provides an auto-generated summary of pyXpcm's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

Profile Classification Model
============================

Creating a PCM
--------------
.. autosummary::
   :toctree: generated/

   pcm

Attributes
----------

.. autosummary::
   :toctree: generated/

   pcm.K
   pcm.F
   pcm.features
   pcm.timeit

Methods
-------

.. autosummary::
   :toctree: generated/

   pcm.fit
   pcm.fit_predict
   pcm.predict
   pcm.predict_proba
   pcm.score
   pcm.bic


Plotting
========

.. autosummary::
    :toctree: generated/

    pcm.plot
    plot.cmap
    plot.colorbar
    plot.subplots
    plot.latlongrid

    plot.scaler
    plot.reducer
    plot.timeit
    plot.preprocessed
    plot.quantile

Datasets module
===============

.. autosummary::
   :toctree: generated/

   datasets.argo
   datasets.isas

Xarray dataset accessor
=======================

.. autosummary::
    :toctree: generated/

    ds_pyxpcm.add
    ds_pyxpcm.clean
    ds_pyxpcm.feature_dict
    ds_pyxpcm.sampling_dim
    ds_pyxpcm.mask
    ds_pyxpcm.quantile
    ds_pyxpcm.robustness
    ds_pyxpcm.robustness_digit
    ds_pyxpcm.fit
    ds_pyxpcm.fit_predict
    ds_pyxpcm.predict
    ds_pyxpcm.predict_proba
    ds_pyxpcm.score
    ds_pyxpcm.bic