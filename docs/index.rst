pyXpcm: Ocean Profile Classification Model
==========================================

**pyXpcm** is a python package to create and work with ocean **Profile Classification Model** that consumes and produces Xarray_ objects. Xarray_ objects are N-D labeled arrays and datasets in Python.

An ocean **Profile Classification Model** allows to automatically assemble ocean profiles in clusters according to their vertical structure similarities.
The geospatial properties of these clusters can be used to address a large variety of oceanographic problems: front detection, water mass identification, natural region contouring (gyres, eddies), reference profile selection for QC validation, etc... The vertical structure of these clusters furthermore provides a highly synthetic representation of large ocean areas that can be used for dimensionality reduction and coherent intercomparisons of ocean data (re)-analysis or simulations.

References
----------

- Maze G. et al. Coherent heat patterns revealed by unsupervised classification of Argo temperature profiles in the North Atlantic Ocean. *Progress in Oceanography* (2017). http://dx.doi.org/10.1016/j.pocean.2016.12.008
- Maze, G., et al. Profile Classification Models. *Mercator Ocean Journal* (2017).
  http://archimer.ifremer.fr/doc/00387/49816
- Jones D. et al. Unsupervised Clustering of Southern Ocean Argo Float Temperature Profiles. *Journal of Geophysical Research: Oceans* (2019). http://dx.doi.org/10.1029/2018JC014629

Pre-trained PCM
---------------
- Maze, G. A Profile Classification Model from North-Atlantic Argo temperature data. *SEANOE Sea scientific open data edition*.
  http://dx.doi.org/10.17882/47106

Documentation
-------------

**Getting Started**

* :doc:`install`
* :doc:`overview`
* :doc:`basic_new.ipynb`


.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started

    install
    overview
    basic_new.ipynb

**Help & reference**

* :doc:`whats-new`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Help & reference

    whats-new

.. _Xarray: http://xarray.pydata.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _Pangeo: http://pangeo.io