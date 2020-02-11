#!/bin/env python
# -*coding: UTF-8 -*-
#
# Test the creation of documentation notebooks
#
# Created 2019-03-26

import os, stat
# import papermill as pm
#
# def test_nb():
#     """Test documentation notebooks"""
#
#     nb_list = {
#                'example':None,
#                'preprocessing':None,
#                'pcm_prop':None,
#                'io':None,
#                'debug_perf':None,
#                }
#
#     for i, nb in enumerate(nb_list):
#        print("Generating %s notebook [%i/%i] ..." % (nb, i+1, len(nb_list) ))
#        nb_in = "%s.ipynb" % nb
#        nb_out = "../%s.ipynb" % nb
#        if nb_list[nb]:
#           pm.execute_notebook(
#              nb_in,
#              nb_out,
#              parameters = nb_list[nb]
#           )
#        else:
#           pm.execute_notebook(
#              nb_in,
#              nb_out,
#           )
#         # Make nteobook read only to make sure we don't modify it:
#        os.chmod(nb_out, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
#
#     print('Done')