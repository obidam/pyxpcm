.. currentmodule:: pyxpcm

What's New
==========

v0.4 (1 Oct. 2019)
------------------

- Break backward compatibility, nearly all API changed

- Enhancements:

    - Multi-features classification

    - Behind the scene full management of ND-Array classification (so that you can classify directly profiles on a latitude/longitude/time grid, not only a collection of profiles already in 2D array)

    - pyXpcm methods are directly accessible through the :class:`xarray.Dataset` accessor namespace ``pyxpcm``

- pyXpcm now consumes xarray/dask objects all along, not only on the user front-end.


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