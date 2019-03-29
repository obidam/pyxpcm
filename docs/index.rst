.. pyxpcm documentation master file, created by
   sphinx-quickstart on Mon Nov  5 19:15:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyXpcm: Ocean Profile Classification Model
==========================================

**pyXpcm** is a python package to create and work with **Profile Classification Model** that consumes and produces [Xarray](https://github.com/pydata/xarray) objects. Xarray objects are N-D labeled arrays and datasets in Python.

A ocean **Profile Classification Model** allows to automatically assemble ocean profiles in clusters according to their vertical structure similarities.
The geospatial properties of these clusters can be used to address a large variety of oceanographic problems: front detection, water mass identification, natural region contouring (gyres, eddies), reference profile selection for QC validation, etc... The vertical structure of these clusters furthermore provides a highly synthetic representation of large ocean areas that can be used for dimensionality reduction and coherent intercomparisons of ocean data (re)-analysis or simulations.


References
----------

- Maze, G., et al. Coherent heat patterns revealed by unsupervised classification of Argo temperature profiles in the North Atlantic Ocean. *Progress in Oceanography*, 151, 275-292 (2017).
  http://dx.doi.org/10.1016/j.pocean.2016.12.008
- Maze, G., et al. Profile Classification Models. *Mercator Ocean Journal*, 55, 48-56 (2017).
  http://archimer.ifremer.fr/doc/00387/49816
- Maze, G. A Profile Classification Model from North-Atlantic Argo temperature data. *SEANOE Sea scientific open data edition*.
  http://doi.org/10.17882/47106


Documentation
-------------

**Getting Started**

* :doc:`overview`
* :doc:`examples`
* :doc:`install`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    overview
    examples
    install

**Help & reference**

* :doc:`whats-new`
* :doc:`api`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & reference

    whats-new
    api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Xarray: http://xarray.pydata.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _Pangeo: http://pangeo.io