Overview: What is a PCM?
========================

A PCM is a Profile Classification Model, an ocean data mining method introduced by Maze et al (2017) to analyse a collection of oceanic profiles.

**pyXpcm** is an Python implementation of the method that comsumes Xarray_ objects (``xarray.Dataset`` and ``xarray.DataArray``), hence the `x`.

With **pyXpcm** you can simply compute a classification model for a collection of profiles stored in an ``xarray.Dataset``.
**pyXpcm** also provides basic statistics and plotting functions to get you started.

.. image:: _static/pcm-natl-logo.png
   :width: 200 px
   :align: center

.. _Xarray: http://xarray.pydata.org