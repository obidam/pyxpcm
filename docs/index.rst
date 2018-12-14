.. pyxpcm documentation master file, created by
   sphinx-quickstart on Mon Nov  5 19:15:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyXpcm: Profile Classification Modelling for Python Xarray
==========================================================

**Profile Classification Modelling** is a scientific analysis approach based on vertical profiles classification that can be used in a variety of oceanographic problems (front detection, water mass identification, natural region contouring, reference profile selection for validation, etc ...).
It is being developed at Ifremer-LOPS in collaboration with IMT-Atlantique since 2015, and has become mature enough (with publication and communications) to be distributed and made publicly available for continuous improvements with a community development.

**pyXpcm** is a package consuming and producing Xarray_ objects. Xarray_ objects are N-D labeled arrays and datasets in Python. In future release, **pyXpcm** will be able to digest very large datasets, following the Pangeo_ initiative.

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
    api-io

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Xarray: http://xarray.pydata.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _Pangeo: http://pangeo.io