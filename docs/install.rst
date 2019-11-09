.. use "install"

Installation
============

Required dependencies
^^^^^^^^^^^^^^^^^^^^^

- Python 3.6
- Xarray 0.12
- Dask 0.16
- Scikit-learn 0.19

Note that Scikit-learn_ is the default statistic backend, but that if Dask_ml_ is installed you can
use it as well (see :doc:`/api`).

For full plotting functionality (see the :ref:`api-plot` API) the following packages are required:

- Matplotlib 3.0 (mandatory)
- Cartopy 0.17 (for some methods only)
- Seaborn 0.9.0 (for some methods only)

Instructions
^^^^^^^^^^^^

For the latest public release:

.. code-block:: text

    pip install pyxpcm

.. _Scikit-learn: https://scikit-learn.org
.. _Dask_ml: https://ml.dask.org


