.. ipython:: python
   :suppress:
   :okexcept:

    import numpy as np
    import xarray as xr
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature


Working with gridded products
=============================

Ocean profiles are not solely taken from a collection of pointwise observations, they can also come from a nicely gridded product. In this case, profiles are assembled along latitude and longitude grids, also possibly along different times. With Xarray_ it is quite simple to manipulate these data to work with PCM.

Load a dataset
--------------

Let's first load a dummy gridded dataset, a sub-domain over the Gulf Stream from the ISAS15_ product:

.. ipython:: python
    :okwarning:

    from pyxpcm import datasets
    ds = datasets.isas('sample_snapshot').load()


Let's look at surface temperature values:

.. ipython:: python

    @savefig examples_isas15_sst_snapshot.png
    ds['TEMP'].isel(depth=0).plot.contourf(levels=np.arange(0,40,2))

Create a PCM
------------

Note that a Profile Classification Model (PCM) can be created independently of any dataset properties using the :class:`pyxpcm.pcm` class constructor.
But to keep things simple, here we will use the dataset vertical axis down to -800m depth and set it to use temperature data directly from the ``TEMP`` variable.

.. ipython:: python

    from pyxpcm.models import pcm

    # Create the pcm
    m = pcm(K=4, features={'TEMP': ds['depth'].where(ds['depth']>=-800, drop=True)})
    m

Fit the PCM on the dataset
--------------------------

Masking
^^^^^^^
In order to create a PCM from temperature profiles of this gridded product, pyXpcm will automatically mask the domain where features are not available on the PCM axis. This is done internally, but you can have access to the mask through the ``mask`` method:

.. ipython:: python

    mask = ds.pyxpcm.mask(m, dim='depth')
    @savefig examples_isas15_mask.png
    mask.plot()


Training
^^^^^^^^
Now that we have a proper collection of profiles and a PCM, we can simply *fit* the classifier:

.. ipython:: python

    ds.pyxpcm.fit(m, dim='depth')

or equivalently:

.. ipython:: python

    m.fit(ds, dim='depth')

This PCM can now be used to classify any ocean temperature profiles, gridded or not.

Classify ocean profiles
-----------------------

They are two methods to classify ocean profiles:

Hard labelling
^^^^^^^^^^^^^^

Once the PCM is trained, i.e. fitted with a training dataset, we can predict classes that profiles from a :class:`xarray.Dataset` belongs to. We can simply classify profile from the dataset that was used to fit the PCM:

.. ipython:: python

    LABELS = ds.pyxpcm.predict(m, dim='depth')
    LABELS

Each profiles is labelled with one of the possible cluster index from 0 to K-1.

A map of labels can then be quickly drawn:

.. ipython:: python

    @savefig examples_isas15_labels_simple.png
    LABELS.plot(cmap=m.plot.cmap(), add_colorbar=False)
    m.plot.colorbar()

Note that here we made use of the :class:`pyxpcm.plot` methods `cmap` and `colorbar` to produce appropriate colors for labels.

Fuzzy classification
^^^^^^^^^^^^^^^^^^^^

Since the PCM classifier we used (GMM) is fuzzy, we can also predict the probabilities for profiles to belong to each of the classes, the so-called posterior probabilities:

.. ipython:: python

    POSTERIORS = ds.pyxpcm.predict_proba(m, dim='depth')
    POSTERIORS

which can then be map like:

.. ipython:: python
    :okwarning:

    @savefig examples_isas15_posteriors.png
    fig, ax = m.plot.subplots(figsize=(20,5))
    for k in m:
        POSTERIORS.sel(pcm_class=k).plot(ax=ax[k])
        ax[k].set_title("Class %i" % k)

Note here that we made use of the following pyXpcm useful tools:

- the :class:`pyxpcm.pcm` instance has an iterator on clusters, so we can use:

.. ipython:: python
    :okwarning:

    for k in m:
        print('This is class ', k)

- the :class:`pyxpcm.plot` has a ``subplots`` method to automatically create a figure with one subplot per class.

Test
----

.. code-block:: python

    from pyxpcm import plot as pcmplot

    # Nicer map:
    proj = ccrs.PlateCarree()
    subplot_kw={'projection': proj, 'extent': np.array([-72,-38,28,51]) + np.array([-0.1,+0.1,-0.1,+0.1])}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=90, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)

    cmap = m.plot.cmap()
    LABELS.plot.pcolormesh(cmap=cmap, transform=proj, vmin=0, vmax=m.K, add_colorbar=False)
    cl = m.plot.colorbar(ax=ax)

    gl = pcmplot.latlongrid(ax, fontsize=8, dx=5)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('LABELS')


So,  nicer map:

.. ipython:: python
    :suppress:
    :okwarning:

    from pyxpcm import plot as pcmplot
    # Nicer map:
    @savefig examples_isas15_labels_map.png
    proj = ccrs.PlateCarree()
    subplot_kw={'projection': proj, 'extent': np.array([-72,-38,28,51]) + np.array([-0.1,+0.1,-0.1,+0.1])}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=90, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)
    cmap = m.plot.cmap()
    LABELS.plot.pcolormesh(cmap=cmap, transform=proj, vmin=0, vmax=m.K, add_colorbar=False)
    cl = m.plot.colorbar(ax=ax)
    gl = pcmplot.latlongrid(ax, fontsize=8, dx=5)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.set_title('LABELS')

Summary
-------

You can look at in the :doc:`/summary` page for an more an overview.

.. _ISAS15: https://doi.org/10.17882/52367
.. _Xarray: http://xarray.pydata.org/en/stable


