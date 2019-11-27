# -*coding: UTF-8 -*-
"""

Create all notebooks used in the documentation

Parametrized notebooks are compiled with papermill

"""

import os, stat
import papermill as pm

nb_list = {
           'example':None,
           # 'preprocessing':None,
           'pcm_prop':None,
           'io':None,
           'debug_perf':None,
           }

for i, nb in enumerate(nb_list):
   print("Generating %s notebook [%i/%i] ..." % (nb, i+1, len(nb_list) ))
   nb_in = "%s.ipynb" % nb
   nb_out = "../%s.ipynb" % nb

   # Remove existing notebook:
   if os.path.exists(nb_out):
       os.remove(nb_out)

   # Generate new notebook:
   if nb_list[nb]:
      pm.execute_notebook(
         nb_in,
         nb_out,
         parameters = nb_list[nb]
      )
   else:
      pm.execute_notebook(
         nb_in,
         nb_out,
      )
    # Make nteobook read only to make sure we don't modify it:
   os.chmod(nb_out, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

print('Done')