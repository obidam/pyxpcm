Use case summaries
==================

With a collection of ocean profiles
-----------------------------------

.. code-block:: python

    from pyxpcm.models import pcm
    from pyxpcm import datasets as pcmdata
    from pyxpcm import stats as pcmstats
    from pyxpcm import plot as pcmplot
    import numpy as np

    # Load a (dummy) dataset:
    ds = pcmdata.load_argo()

    # Model creation and fit:
    m = pcm(K=3, feature_axis=np.arange(-500, 0, 2), feature_name='temperature')
    m.fit(ds, feature={'temperature': 'TEMP'})

    # Hard/Fuzzy classify data:
    m.predict(ds, feature={'temperature': 'TEMP'}, inplace=True)
    m.predict_proba(ds, feature={'temperature': 'TEMP'}, inplace=True)

    # Compute statistics (quantiles):
    ds = ds.compute()
    pcmstats.quant(ds, of='TEMP', using='PCM_LABELS', name='TEMP_Q', inplace=True)
    pcmstats.quant(ds, of='PSAL', using='PCM_LABELS', name='PSAL_Q', inplace=True)

    # Plots:
    pcmplot.scaler(m)
    pcmplot.quant(m, ds['TEMP_Q'])
    pcmplot.quant(m, ds['PSAL_Q'], xlim=[36, 37])


With a gridded collection of ocean profiles
-------------------------------------------

.. code-block:: python

    from pyxpcm.models import pcm
    from pyxpcm import datasets as pcmdata
    from pyxpcm import stats as pcmstats
    from pyxpcm import plot as pcmplot
    import numpy as np

    # Load a gridded (dummy) dataset:
    ds = pcmdata.load_isas15()
    ds['depth'] = -np.abs(ds['depth']) # Make sure depth is negative defined

    # Mask of profiles to classify:
    zmin = -300.
    ds['mask'] = np.bitwise_and( ~np.isnan(ds['TEMP'].isel(depth=0)), \
                                (ds['TEMP'].where(ds['depth']>=zmin).notnull().sum(dim='depth') == \
                                     len(np.where(ds['depth']>=zmin)[0])))
    # Create a 2D array to work with:
    dsub = ds.stack(n_samples=('latitude', 'longitude')).transpose('n_samples', 'depth')
    dsub = dsub.where(dsub.mask == 1, drop=True)

    # Set the PCM vertical axis from the dataset:
    feature_axis=ds['depth'].where(ds['depth']>=zmin, drop=True)

    # models creation:
    m = pcm(K=4, feature_axis=feature_axis, feature_name='TEMP')

    # Fit, predict:
    m.fit_predict(dsub, inplace=True)

    # Prob:
    m.predict_proba(dsub, inplace=True, classdimname='class')

    # Quantiles for typical class profiles:
    dsub = dsub.compute()
    pcmstats.quant(dsub, of='TEMP', using='PCM_LABELS', name='TEMP_Q', inplace=True)

    # Get back to a lat/lon grid:
    dsub = dsub.unstack('n_samples')

    # Possibly add PCM new variables to the initial dataset:
    ds_new = ds.combine_first(dsub)
    # print(ds_new)

    # Plot map of labels:
    ds_new['PCM_LABELS'].plot(cmap=m.plot.cmap(), add_colorbar=False)
    m.plot.colorbar()

    # Plot map of posteriors (probability of each profile to belong to each classes):
    ds_new['PCM_POST'].plot(x='longitude', y='latitude', col='class', col_wrap=2)

    # Plot typical profiles:
    ds_new['TEMP_Q'].plot(y='depth', hue='quantile', col='class', col_wrap=2)

    plt.figure()
    ds_new['TEMP_Q'].sel(quantile=[0.05,0.95]).diff('quantile').plot(y='depth', hue='class')
