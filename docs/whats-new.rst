.. currentmodule:: pyxpcm

What's New
==========

Upcoming release
----------------

- Fix bug with last numpy and integer management :issue:`42`.

v0.4.1 (21 Feb. 2020)
---------------------

- Improved documentation

- Improved unit testing

- Bug fix:
    -  Fix a bug in the preprocessing step using dask_ml bakend that would cause an error for data already in dask arrays

v0.4.0 (1 Nov. 2019)
--------------------

.. warning::

    The API has changed, break backward compatibility.

- Enhancements:

    - Multiple-features classification

    - ND-Array classification (so that you can classify directly profiles from gridded products, eg: latitude/longitude/time grid, and not only a collection of profiles already in 2D array)

    - pyXpcm methods can be accessed through the :class:`xarray.Dataset` accessor namespace ``pyxpcm``

    - Allow to choose statistic backends (sklearn, dask_ml or user-defined)

    - Save/load PCM to/from netcdf files

- pyXpcm now consumes xarray/dask objects all along, not only on the user front-end. This add a small overhead with small dataset but allows for PCM to handle large and more complex datasets.


v0.3 (5 Apr. 2019)
------------------

- Removed support for python 2.7

- Added more data input consistency checks

- Fix bug in interpolation and plotting methods

- Added custom colormap and colorbar to plot module

v0.2 (26 Mar. 2019)
-------------------

- Upgrade to python 3.6 (compatible 2.7)

- Added test for continuous coverage

- Added score and bic methods

- Improved vocabulary consistency in methods

v0.1.3 (12 Nov. 2018)
---------------------

- Initial release.