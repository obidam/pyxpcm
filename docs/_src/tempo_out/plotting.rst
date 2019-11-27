Plotting
========

pyXpcm comes with basic plotting tools to help you visaulise your classified data: these are available in the
:class:`pyxpcm.plot` module.

To begin with, import libraries and fit a PCM on dummy data:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pyxpcm.models import pcm
    from pyxpcm import datasets as pcmdata
    from pyxpcm import plot as pcmplot
    from pyxpcm import stats as pcmstats

    # Load data
    ds = pcmdata.load_argo()

    # Model creation and fit:
    m = pcm(K=3, feature_axis=np.arange(-80, 0, 2), feature_name='TEMP')
    m.fit(ds)

    # Hard/Fuzzy classify data:
    m.predict(ds, inplace=True)
    m.predict_proba(ds, inplace=True)

    # Compute statistics (quantiles):
    ds = ds.compute() # This is necessary if data are in dask arrays
    pcmstats.quant(ds, of='TEMP', using='PCM_LABELS', q=[0.05, 0.5, 0.95], name='TEMP_Q', inplace=True)


.. ipython:: python
    :suppress:
    :okwarning:

    import numpy as np
    import matplotlib.pyplot as plt
    from pyxpcm.models import pcm
    from pyxpcm import datasets as pcmdata
    from pyxpcm import plot as pcmplot
    from pyxpcm import stats as pcmstats
    ds = pcmdata.load_argo()
    m = pcm(K=3, feature_axis=np.arange(-80, 0, 2), feature_name='TEMP')
    m.fit(ds)
    m.predict(ds, inplace=True)
    m.predict_proba(ds, inplace=True)
    ds = ds.compute() # This is necessary if data are in dask arrays
    pcmstats.quant(ds, of='TEMP', using='PCM_LABELS', q=[0.05, 0.5, 0.95], name='TEMP_Q', inplace=True)


Quantiles
---------

In order to look at the structure of the classes, we can use quantile variables (see :doc:`/statistics` in order to get more details on computing class-based quantiles).

.. ipython:: python
    :okwarning:

    @savefig examples_quantiles.png width=75%
    pcmplot.quant(m, ds['TEMP_Q'])

The spread of each class can then be computed and plotted like:

.. ipython:: python
    :okwarning:

    spread = ds['TEMP_Q'].sel(quantile=[0.05, 0.95]).diff('quantile')

    @savefig examples_quantiles_xr2.png
    spread.plot(y='DEPTH', hue='pcm_class')

Profiles of a class
-------------------

To overlay all profiles attributed to a given class one can use:

.. ipython:: python
    :okwarning:

    @savefig examples_profiles_per_class.png width=100%
    fig, axes = plt.subplots(ncols=m.K, figsize=(15,6), sharex='col', sharey='row')
    for k in m:
        ds['TEMP'].where(ds['PCM_LABELS']==k, drop=True).plot(ax=axes[k], y='DEPTH', hue='N_PROF', add_legend=False)
        axes[k].grid(True); axes[k].set_title(('Profiles in class %i')%(k))


PCM scaler properties
---------------------

It is possible to plot the PCM scaler mean and std:

.. ipython:: python
    :okexcept:
    :okwarning:

    @savefig examples_scaler.png width=75%
    pcmplot.scaler(m)

PCM colormap and colorbar
-------------------------

A :class:`pyxpcm.pcm.plot` class instance has a colormap method to return a LinearSegmentedColormap matplotlib colormap:

.. ipython:: python
    :okwarning:

    cmap = m.plot.cmap()
    ax = plt.subplot(1,1,1); ax.axis("off")
    @savefig examples_colormap.png width=200px height=20px
    plt.imshow(np.outer(np.ones(10), np.arange(0,1,0.01)), aspect='auto', cmap=cmap, origin="lower")


Misc
----

One can also simply use the Xarray_ plotting capabilities:

.. ipython:: python
    :okwarning:

    @savefig examples_quantiles_xr1.png width=100%
    g = ds['TEMP_Q'].plot(y='DEPTH', hue='quantile', col='pcm_class', col_wrap=3)
    for i, ax in enumerate(g.axes.flat): ax.grid(True)


.. _Xarray: http://xarray.pydata.org