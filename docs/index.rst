pyXpcm: Ocean Profile Classification Model
==========================================

**pyXpcm** is a python package to create and work with ocean **Profile Classification Model** that consumes and produces Xarray_ objects. Xarray_ objects are N-D labeled arrays and datasets in Python.

An ocean **Profile Classification Model** allows to automatically assemble ocean profiles in clusters according to their vertical structure similarities.
The geospatial properties of these clusters can be used to address a large variety of oceanographic problems: front detection, water mass identification, natural region contouring (gyres, eddies), reference profile selection for QC validation, etc... The vertical structure of these clusters furthermore provides a highly synthetic representation of large ocean areas that can be used for dimensionality reduction and coherent intercomparisons of ocean data (re)-analysis or simulations.


Documentation
-------------

**Getting Started**

* :doc:`overview`
* :doc:`install`
* :doc:`model_catalogue`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    overview
    install
    model_catalogue

**User guide**

* :doc:`example`
* :doc:`pcm_prop`
* :doc:`io`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: User guide

    example.ipynb
    pcm_prop.ipynb
    io.ipynb

**Advanced**

* :doc:`preprocessing`
* :doc:`debug_perf`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Advanced

    preprocessing.ipynb
    debug_perf.ipynb

**Help & reference**

* :doc:`bibliography`
* :doc:`whats-new`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & reference

    bibliography
    whats-new
    api

.. _Xarray: http://xarray.pydata.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _Pangeo: http://pangeo.io