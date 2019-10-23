.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

.. currentmodule:: pyxpcm

.. autosummary::
    :toctree: generated/

    models.pcm
    models.pcm.K
    models.pcm.F
    models.pcm.features
    models.pcm.timeit
    models.pcm.fit
    models.pcm.fit_predict
    models.pcm.predict
    models.pcm.predict_proba
    models.pcm.score
    models.pcm.bic
    models.pcm.to_netcdf

    models.pcm.ravel
    models.pcm.unravel

    models.pcm.plot
    models.pcm.plot.cmap
    models.pcm.plot.colorbar
    models.pcm.plot.subplots
    models.pcm.plot.scaler
    models.pcm.plot.reducer
    models.pcm.plot.timeit
    models.pcm.plot.preprocessed
    models.pcm.plot.quantile

    plot.cmap
    plot.latlongrid
    plot.scaler
    plot.reducer
    plot.quantile

    tutorial.open_dataset

    models.pcm.to_netcdf
    io.to_netcdf
    io.open_netcdf

    pyxpcm.xarray.pyXpcmDataSetAccesso
    pyxpcm.xarray.pyXpcmDataSetAccesso.add
    pyxpcm.xarray.pyXpcmDataSetAccesso.clean
    pyxpcm.xarray.pyXpcmDataSetAccesso.feature_dict
    pyxpcm.xarray.pyXpcmDataSetAccesso.sampling_dim
    pyxpcm.xarray.pyXpcmDataSetAccesso.mask
