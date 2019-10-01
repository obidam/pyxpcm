.. currentmodule:: pyxpcm

#############
API reference
#############

This page provides an auto-generated summary of pyXpcm's API. For more details and examples, refer to the relevant chapters in the main part of the documentation.

Top-level PCM functions
=======================

Creating a PCM
--------------
.. autosummary::
   :toctree: generated/

   pcm

Attributes
--------------

.. autosummary::
   :toctree: generated/

   pcm.K
   pcm.F

Contents
------------

.. autosummary::
   :toctree: generated/

   pcm.features
   pcm.timeit

Computation
---------------

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

Tools
-----
.. autosummary::
    :toctree: generated/

    plot.cmap
    plot.colorbar
    plot.subplots
    plot.latlongrid

Plot PCM Contents
-----------------
.. autosummary::
    :toctree: generated/

    plot.scaler
    plot.reducer
    plot.timeit
    plot.preprocessed
    plot.quantile

Helper
======

.. autosummary::
   :toctree: generated/

    tutorial.open_dataset

Xarray dataset accessor: the **pyxpcm** name space
==================================================


PCM Computation
---------------

.. autosummary::
    :toctree: generated/

    xr_pyxpcm.fit
    xr_pyxpcm.fit_predict
    xr_pyxpcm.predict
    xr_pyxpcm.predict_proba
    xr_pyxpcm.score
    xr_pyxpcm.bic

Diagnostics
-----------

.. autosummary::
    :toctree: generated/

    xr_pyxpcm.quantile
    xr_pyxpcm.robustness
    xr_pyxpcm.robustness_digit


Low-level functions
-------------------

.. autosummary::
    :toctree: generated/

    xr_pyxpcm.add
    xr_pyxpcm.clean
    xr_pyxpcm.feature_dict
    xr_pyxpcm.sampling_dim
    xr_pyxpcm.mask