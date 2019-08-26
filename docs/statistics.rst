Statistics
==========

pyXpcm comes with useful statistic tool to post-process and analyse your classified data: these are available in the
:class:`pyxpcm.stats` module.

Note that because the classifier works in a reduced space of scaled and interpolated data, it is quite complex to visualise clusters
in their original space. On the other hand, once profiles are classified, one can compute statistics in the original data space based on classification metrics.

To begin with, import libraries and fit a PCM on dummy data (see also use case summaries):

.. code-block:: python

    import numpy as np
    from pyxpcm.pcmodel import pcm
    from pyxpcm import datasets as pcmdata
    from pyxpcm import stats as pcmstats

    # Load data
    ds = pcmdata.load_argo()

    # Model creation and fit:
    m = pcm(K=3, feature_axis=np.arange(-500, 0, 2), feature_name='temperature')
    m.fit(ds, feature={'temperature': 'TEMP'})

    # Hard/Fuzzy classify data:
    m.predict(ds, feature={'temperature': 'TEMP'}, inplace=True)
    m.predict_proba(ds, feature={'temperature': 'TEMP'}, inplace=True)

.. ipython:: python
    :suppress:
    :okwarning:

    import numpy as np
    from pyxpcm.pcmodel import pcm
    from pyxpcm import datasets as pcmdata
    from pyxpcm import stats as pcmstats
    ds = pcmdata.load_argo()
    m = pcm(K=3, feature_axis=np.arange(-500, 0, 2), feature_name='temperature')
    m.fit(ds, feature={'temperature': 'TEMP'})
    m.predict(ds, feature={'temperature': 'TEMP'}, inplace=True)
    m.predict_proba(ds, feature={'temperature': 'TEMP'}, inplace=True)


Quantiles
---------

:class:`pyxpcm.stats` provide a quantiles computation method that make full leverage of Xarray_.

For instance, in order to look at the typical structure of a class, one can compute quantiles of the classified variable. This can be done like:

.. ipython:: python
    :okwarning:

    ds = ds.compute() # This is necessary if data are in dask arrays
    pcmstats.quant(ds, of='TEMP', using='PCM_LABELS', q=[0.05, 0.5, 0.95], name='TEMP_Q')
    ds

.. _Xarray: http://xarray.pydata.org/en/stable/
