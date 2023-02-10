
# Determine backends to test:
backends = list()
try:
    import sklearn
    backends.append('sklearn')
except ModuleNotFoundError:
    pass

try:
    import dask_ml
    backends = ['dask_ml']
except ModuleNotFoundError:
    pass

backends_ids = ["backend=%s" % s for s in backends]