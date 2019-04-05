.. ipython:: python
   :suppress:

    import numpy as np
    import xarray as xr
    import cartopy.crs as crs


Working with gridded products
=============================

Ocean profiles are not solely taken from observations, they can also come from a nicely gridded product. In this case, profiles are assembled along latitude and longitude grids, also possibly along different times. With Xarray_ it is quite simple to manipulate these data to work with PCM.

At this point, it is on the user shoulder to pre/post process its gridded data in order to work with PCM. However, in a future release, we'll try to reduce the overhead of manipulating the grid by incorporating all the machinery directly within the library `(see this issue) <https://github.com/obidam/pyxpcm/issues/6>`_. This is a top priority to us.

In the mean time, to get you started, we illustrate below how to work with PCM on a gridded product.

Load a dataset
--------------

Let's first load a dummy gridded dataset, a sub-domain over the Gulf Stream from the ISAS15_ product:

.. ipython:: python

    from pyxpcm import datasets as pcmdata
    ds = pcmdata.load_isas15()
    ds

.. ipython:: python
   :suppress:

    ds['depth'] = -ds['depth']

Let's look at the surface temperature values:

.. ipython:: python

    @savefig examples_isas15_sst_snapshot.png
    ds['TEMP'].isel(depth=0).plot.contourf(levels=np.arange(0,40,2))

Create a PCM
------------

Note that a Profile Classification Model (PCM) can be created independently of any dataset properties using the :class:`pyxpcm.pcm` class constructor.
But to keep things simple, here we will use the dataset vertical axis down to -800m depth and set it to use temperature data directly from the ``TEMP`` variable.

.. ipython:: python

    from pyxpcm.pcmodel import pcm

    # Define the feature axis we want to use for classification:
    zmin = -800
    feature_axis = ds['depth'].where(ds['depth']>=zmin, drop=True)

    # Create the pcm
    m = pcm(K=4, feature_axis=feature_axis, feature_name='TEMP')
    m

Fit the PCM on the dataset
--------------------------

Masking
^^^^^^^
In order to create a PCM from temperature profiles of this gridded product, we first need to determine the domain of analysis where profiles will be plain, i.e. will not contain any NaN. This will depend on the maximum depth of profiles to analyse.

.. ipython:: python

    ds['mask'] = np.bitwise_and( \
                    ~np.isnan(ds['TEMP'].isel(depth=0)), \
                    (ds['TEMP'].where(ds['depth']>=zmin).notnull().sum(dim='depth') == \
                                     len(np.where(ds['depth']>=zmin)[0])))

    ax = plt.axes(projection=crs.PlateCarree())
    ds['mask'].plot.contourf(levels=3, transform=crs.PlateCarree())
    @savefig examples_isas15_mask.png
    ax.set_extent([-80,-30,25,55]); ax.coastlines(); ax.gridlines(); ax.set_title('PCM Mask')


With this mask, we can easily select all temperature profiles reaching at least -800m depth.

Let's now assemble the collection of plain profiles to be classified with the PCM:

.. ipython:: python

    dsub = ds.stack(n_samples=('latitude', 'longitude')).transpose('n_samples', 'depth')
    dsub = dsub.where(dsub.mask == 1, drop=True)
    dsub

We used the :func:`xarray.Dataset.stack` method to create a [sample x feature] 2-dimensional array to be used in PCM.

Training
^^^^^^^^

Now that we have a proper collection of profiles and a PCM, we can simply *fit* the classifier:

.. ipython:: python

    m.fit(dsub)

This PCM can now be used to classify any ocean profiles.

Classify ocean profiles
-----------------------

There are two methods to then classify ocean profiles:

Hard labelling
^^^^^^^^^^^^^^

Once the PCM is trained, i.e. fitted with a training dataset, we can predict classes that profiles from a :class:`xarray.Dataset` belongs to. We can simply classify profile from the dummy dataset that was used to fit the PCM:

.. ipython:: python

    LABELS = m.predict(dsub)
    LABELS = LABELS.unstack('n_samples')
    LABELS

Each profiles is labelled with one of the possible cluster index from 0 to K-1. The output ``labels`` is a :class:`xarray.DataArray` that can simply be unstacked to get back to the original dataset lat/lon grid.

A map of labels can then be drawn:

.. ipython:: python

    ax = plt.axes(projection=crs.PlateCarree())
    LABELS.plot(cmap=m.plot.cmap(), transform=crs.PlateCarree(), add_colorbar=False)
    m.plot.colorbar()
    @savefig examples_isas15_labels.png
    ax.set_extent([-80,-30,25,55]); ax.coastlines(); ax.gridlines(); ax.set_title('PCM Labels')

Note that here we made use of the :class:`pyxpcm.plot` methods `cmap` and `colorbar` to produce appropriate colors for labels.

Fuzzy classification
^^^^^^^^^^^^^^^^^^^^

Since the PCM classifier we used (GMM) is fuzzy, we can also predict the probabilities for profiles to belong to each of the classes, the so-called posterior probabilities:

.. ipython:: python

    POSTERIORS = m.predict_proba(dsub).unstack('n_samples')
    POSTERIORS

which can then be map like:

.. ipython:: python
    :okwarning:

    g = POSTERIORS.plot(x='longitude', y='latitude', col='N_CLASS', col_wrap=2, \
                                transform=crs.PlateCarree(), subplot_kws={'projection':crs.PlateCarree()},\
                                 aspect=2, size=3)
    @savefig examples_isas15_posteriors.png
    for i, ax in enumerate(g.axes.flat):
        ax.set_extent([-80,-30,25,55])
        ax.coastlines()
        ax.gridlines()

Summary
-------

You can look at in the :doc:`/summary` page for an more an overview.

.. _ISAS15: https://doi.org/10.17882/52367
.. _Xarray: http://xarray.pydata.org/en/stable


