.. _usage_matlab:

Usage
=====

Here are some typical use cases.

.. contents::
   :local:
   :depth: 1

Train a PCM on your data
------------------------

Let's create a dummy dataset: an array 'temp' of dimension [N_LEVEL, N_SAMPLE] define along a vertical axis 'dpt'.

This could be any collection of vertical profiles, as long as it is a 2D plain matrix, without NaNs.

Also note that the toolbox works with vertical axis with negative values and oriented from the surface toward the bottom.

.. code-block:: matlab

    % Dummy dataset:
    N_SAMPLE = 1000;
    N_LEVEL = 50;
    temp = sort(rand(N_LEVEL, N_SAMPLE)*10+10, 1, 'descend');
    dpt = linspace(0, -1000, N_LEVEL)';

    % Plot one random profile with:
    plot(temp(:, randi([1, N_SAMPLE],1)),dpt)

Now we can train a PCM on this data, using the same vertical axis:

.. code-block:: matlab

    K = 4; % Number of class to fit
    PCM = pcmtrain(temp, K, 'full', dpt, 'maxvar',inf);

The parameter 'full' defines the classification model covariance matrix shape.

The parameter 'maxvar'=inf is the maximum variance to be retained during the compression step, it is an option and is
used here only because we have random data hard to reduce.

The trained Profile Classification Model is in the PCM structure where all properties should be quite self-explanatory.
But no direct use of the PCM property should be done.

.. code-block:: matlab

    >> PCM
    PCM =
      struct with fields:
             DPTmodel: [501 double]
                 EOFs: [5050 double]
                    K: 4
                  LLH: 13634.1879135846
                   Np: 1000
                    V: [150 double]
                X_ave: [501 double]
                X_ref: [501 double]
                X_std: [501 double]
            covarTYPE: 'full'
             doREDUCE: 1
               maxvar: 100
                  mix: [11 struct]
        normalization: 1
               readme: 'This PCM was created using pcmtrain.m'

Now you can classify the dataset:

.. code-block:: matlab

    [POST LABEL] = pcmpredict(PCM, dpt, temp)

The 'POST' array is the class component probability for each data, 'LABEL' is the class maximising 'POST' for each data.


Saving and loading a PCM
------------------------
The toolbox comes with handy functions to save and load the PCM in a netcdf format.

To save a PCM:

.. code-block:: matlab

    pcmsave('my_first_pcm.nc', PCM);

To load PCM:

.. code-block:: matlab

    PCM = pcmload('my_first_pcm.nc');

The North-Atlantic PCM trained on Argo data used in Maze et al (Pr.Oc., 2017) is published here:
`http://doi.org/10.17882/47106 <http://doi.org/10.17882/47106>`_


Classify a profile with an existing PCM
---------------------------------------

It is important to note that the PCM has its own vertical axis, so that any new data defined on another vertical axis can
be predicted with a PCM that was possibility train on another dataset:

.. code-block:: matlab

    % Load a profile classification model structure:
    % (this netcdf file can be downloaded here: http://doi.org/10.17882/47106)
    PCM = pcmload('Argo-NATL-PCM-model.nc');
    % This PCM is defined on the 0-1405m depth range.

    % Load an Argo profile (using the LOPS-Argo Matlab library):
    [Co,Dim] = libargo.read_netcdf_allthefile('6900828_prof.nc');
    pres = Co.pres_adjusted.data(10,:);
    dpt  = -(pres(:)*1.019716); % % Convert pressure to depth (rough approximation for demonstration purpose only)
    temp = Co.temp_adjusted.data(10,:)';

    % Classify the new Argo profile:
    [POST LABEL] = pcmpredict(PCM, dpt, temp) % Note that dpt and temp MUST be (z,:)

Note that this example uses the LOPS-Argo Matlab toolbox to load Argo netcdf data. **The PCM toolbox does not depend on it, this
is only for examples**. The toolbox is available here:

.. code-block:: text

    svn checkout https://forge.ifremer.fr/svn/lpoargo/projects/matlab forge-lpoargo

and add it to your path:

.. code-block:: matlab

    addpath('forge-lpoargo');
