Overview
========

What is an ocean PCM?
---------------------

An ocean PCM is a **Profile Classification Model for ocean data**, a statistical procedure to classify ocean vertical profiles into a finite set of "clusters".
Depending on the dataset, such clusters can show space/time coherence that can be used in many different ways to study the ocean.

Statistic method
----------------

It consists in conducting **un-supervised classification** `(clustering) <https://en.wikipedia.org/wiki/Cluster_analysis>`_ with vertical profiles of one or more ocean variables.

Each levels of the vertical axis of each ocean variables are considered a **feature**.
One ocean vertical profile with ocean variables is considered a **sample**.

All the details of the Profile Classification Modelling (PCM) statistical methodology can be found in `Maze et al, 2017`_.

Illustration
------------

Given a collection of Argo temperature profiles in the North Atlantic, a PCM analysis is applied and produces an optimal set of 8 ocean temperature profile classes. The PCM clusters synthesize the structural information of heat distribution in the North Atlantic. Each clusters objectively define an ocean region where dynamic gives rise to an unique vertical stratification pattern.

.. image:: _static/graphical-abstract.png
   :width: 100%
   :align: center

`Maze et al, 2017`_ applied it to the North Atlantic with Argo temperature data. `Jones et al, 2019`_, later applied it to the Southern Ocean, also with Argo temperature data. Rosso et al (in prep) has applied it to the Southern Indian Ocean using both temperature and salinity Argo data.


pyXpcm
------

**pyXpcm** is an Python implementation of the PCM method that consumes and produces Xarray_ objects (:class:`xarray.Dataset` and :class:`xarray.DataArray`), hence the `x`.

With **pyXpcm** you can conduct a PCM analysis for a collection of profiles (gridded or not), of one or more ocean variables, stored in an :class:`xarray.Dataset`.
**pyXpcm** also provides basic statistics and plotting functions to get you started with your analysis.

The philosophy of the **pyXpcm** toolbox is to create and be able to use a PCM from and on different ocean datasets and variables. In order to achieve this, a PCM is created with information about ocean variables to classify and the vertical axis of these variables. Then this PCM can be fitted and subsequently classify ocean profiles from any datasets, as long as it contains the PCM variables.

The **pyXpcm** procedure is to preprocess (stack, scale, reduce and combine data) and then to fit a classifier on data. Once the model is fitted **pyXpcm** can classify data. The library uses many language and logic from `Scikit-learn`_ but doesn't inherit from a :class:`sklearn.BaseEstimator`.

.. _Scikit-learn: https://scikit-learn.org/
.. _Xarray: http://xarray.pydata.org
.. _Maze et al, 2017: http://dx.doi.org/10.1016/j.pocean.2016.12.008
.. _Jones et al, 2019: http://dx.doi.org/10.1029/2018jc014629
