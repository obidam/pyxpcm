.. _examples:

Examples
========

To get you started with examples, first load a sample dataset of Argo profiles interpolated on standard depth levels:

.. ipython:: python

    from pyxpcm import datasets as pcmdata
    ds = pcmdata.load_argo()
    ds

We see that this dataset has 100 profiles with 21 depth levels.