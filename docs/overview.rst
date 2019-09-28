Overview
========

What is an ocean PCM?
---------------------

An ocean PCM is a **Profile Classification Model for ocean data**, a statistical procedure to aggregate ocean vertical profiles into a finite set of "clusters".
Depending on the dataset, such clusters can show space/time coherence that can be used in many different ways to study the ocean.

The method was introduced by `Maze et al (2017)`_ and applied to the North Atlantic. `Jones et al (2019)`_ later applied it
to the Southern Ocean.

pyXpcm
------

**pyXpcm** is an Python implementation of the PCM method that comsumes and produces Xarray_ objects (:class:`xarray.Dataset` and :class:`xarray.DataArray`), hence the `x`.

With **pyXpcm** you can simply compute a classification model for a collection of profiles stored in an :class:`xarray.Dataset`.
**pyXpcm** also provides basic statistics and plotting functions to get you started.

The philosophy of the **pyXpcm** toolbox is to create and be able to use a PCM from and on different ocean datasets and variables. In order to achieve this, a PCM is created with information about ocean variables to classify and the vertical axis of these variables. Then this PCM can be fitted and subsequently classify ocean profiles from any datasets, as long as it contains the PCM variables.

The **pyXpcm** procedure is to preprocess (interpolate, scale, reduce) and then fit or classify data. It uses many language and logic from `scikit-learn scikit_` but doesn't inherit from a :class:`sklearn.BaseEstimator`.

Illustration
------------

Figure below is from `Maze et al (2017)`_. Given a collection of Argo profiles in the North Atlantic, the PCM procedure is applied and produces an optimal set of 8 ocean temperature profile clusters. The PCM clusters synthesize the structural information of heat distribution in the North Atlantic. Each clusters objectively define an ocean region where dynamic gives rise to an unique vertical stratification pattern.

.. image:: _static/graphical-abstract.png
   :width: 100%
   :align: center

.. _scikit: https://scikit-learn.org/
.. _Xarray: http://xarray.pydata.org
.. _Jones et al (2019): http://dx.doi.org/10.1029/2018jc014629
.. _Maze et al (2017): http://dx.doi.org/10.1016/j.pocean.2016.12.008