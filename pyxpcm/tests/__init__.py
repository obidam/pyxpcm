# -*coding: UTF-8 -*-
"""

Test suite for pyXpcm continuous integration

#todo Fix Travis environment failing on io test because of missing dask.preprocessing backend
eg: https://travis-ci.org/gmaze/pyxpcm/builds/599093644

"""


# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl

    # Order of imports is important here.
    # Using a different backend makes Travis CI work
    mpl.use("Agg")
except ImportError:
    pass

import xarray
print("xarray: %s, %s" % (xarray.__version__, xarray.__file__))

import sklearn
print("sklearn: %s, %s" % (sklearn.__version__, sklearn.__file__))