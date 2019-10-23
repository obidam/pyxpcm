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
   pyxpcm.load_netcdf

Attributes
--------------

.. autosummary::
   :toctree: generated/

   pcm.K
   pcm.F
   pcm.features

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


Low-level PCM properties and functions
======================================

.. autosummary::
   :toctree: generated/

   pcm.timeit
   pcm.ravel
   pcm.unravel


.. _api-plot:

Plotting
========

.. autosummary::
    :toctree: generated/

    pcm.plot

Plot PCM Contents
-----------------
.. autosummary::
    :toctree: generated/

    plot.quantile
    plot.scaler
    plot.reducer
    plot.preprocessed
    plot.timeit

Tools
-----
.. autosummary::
    :toctree: generated/

    plot.cmap
    plot.colorbar
    plot.subplots
    plot.latlongrid

Save/load PCM models
====================

.. autosummary::
    :toctree: generated/

    pcm.to_netcdf
    pyxpcm.load_netcdf

Helper
======

.. autosummary::
   :toctree: generated/

    tutorial.open_dataset

Xarray *pyxpcm* name space
==========================

.. automodule:: pyxpcm.xarray

.. autoclass:: pyXpcmDataSetAccessor()
    :members: