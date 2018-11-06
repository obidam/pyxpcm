.. _installation:

Installation
============

Get the source code
^^^^^^^^^^^^^^^^^^^
Clone the last version of the toolbox from the git repository:

.. code-block:: text

    git clone https://github.com/obidam/pcm.git

Dependencies
^^^^^^^^^^^^
The PCM Matlab toolbox relies on the
`Netlab Neural Network Software, version 3.3 <http://www.aston.ac.uk/eas/research/groups/ncrg/resources/netlab>`_.
It is included in the PCM Matlab toolbox under the `lib` folder, so you don't have to install it.

Install on the Matlab path
^^^^^^^^^^^^^^^^^^^^^^^^^^
Simply ensure that you have the PCM Matlab toolbox (`src`) and its dependencies (`lib`) folders on your Matlab path !
This can simply be done by adding the following lines to your startup file:

.. code-block:: matlab

    addpath(fullfile('pcm','matlab','src'));
    addpath(fullfile('pcm','matlab','lib','netlab3_3'));