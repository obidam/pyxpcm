pyXpcm: Ocean Profile Classification Model
==========================================

[![DOI-Software](https://img.shields.io/badge/DOI%3A%20Software-10.5281%2Fzenodo.3906236-blue)](//dx.doi.org/10.5281/zenodo.3906236)
[![DOI-Paper](https://img.shields.io/badge/DOI%3A%20Paper-10.1016%2Fj.pocean.2016.12.008-blue)](//dx.doi.org/10.1016/j.pocean.2016.12.008)  
[![Documentation Status](https://img.shields.io/readthedocs/pyxpcm?logo=readthedocs)](https://pyxpcm.readthedocs.io/en/latest/?badge=latest)
[![CI tests](https://github.com/obidam/pyxpcm/actions/workflows/tests.yml/badge.svg)](https://github.com/obidam/pyxpcm/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/obidam/pyxpcm/branch/master/graph/badge.svg)](https://codecov.io/gh/obidam/pyxpcm)
[![PyPI](https://img.shields.io/pypi/v/pyxpcm)](//pypi.org/project/pyxpcm/)  
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) 

**pyXpcm** is a python package to create and work with ocean **Profile Classification Model** that consumes and produces [Xarray](https://github.com/pydata/xarray) objects. Xarray objects are N-D labeled arrays and datasets in Python. 

A ocean **Profile Classification Model** allows to automatically assemble ocean profiles in clusters according to their vertical structure similarities.   
The geospatial properties of these clusters can be used to address a large variety of oceanographic problems: front detection, water mass identification, natural region contouring (gyres, eddies), reference profile selection for QC validation, etc... The vertical structure of these clusters furthermore provides a highly synthetic representation of large ocean areas that can be used for dimensionality reduction and coherent intercomparisons of ocean data (re)-analysis or simulations.   

## Why pyXpcm?
The **Ocean dynamics** and its 3-dimensional structure and variability is so complex that it is very difficult to develop objective and efficient diagnostics of horizontally and vertically coherent oceanic patterns. However, identifying such **patterns** is crucial to the understanding of interior mechanisms as, for instance, the integrand giving rise to Global Ocean Indicators (e.g. heat content and sea level rise). We believe that, by using state-of-the-art **machine learning** algorithms and by building on the increasing availability of ever-larger **in situ and numerical model datasets**, we can address this challenge in a way that was simply not possible a few years ago. Following this approach, **Profile Classification Modelling** focuses on the smart identification of vertically coherent patterns and their space/time distribution and occurrence.

## Documentation
[https://pyxpcm.readthedocs.io](https://pyxpcm.readthedocs.io)

## Install

Latest release:

    pip install pyxpcm
    
Development version:

    pip install git+https://github.com/obidam/pyxpcm.git
    
## Dev roadmap
This library has shown to be quiet robust over a couple of years now: it kept working without active dev and was used in [numerous research papers](https://pyxpcm.readthedocs.io/en/latest/bibliography.html) as well to create new tools in big projects like [Blue-Cloud](https://blue-cloud.org/article/applying-machine-learning-methods-ocean-patterns-and-ocean-regimes-indicators) and Euro-Argo RISE (see [here for Argo floats QC](10.5281/zenodo.7362293) and [here for boundary currents monitoring](https://github.com/euroargodev/boundary_currents_pcm/)). 

Our plan is a new release in 2023 that will:

- [ ] fix issues: [#33](https://github.com/obidam/pyxpcm/issues/33), [#38](https://github.com/obidam/pyxpcm/issues/38)
- [ ] implement new features: [#5](https://github.com/obidam/pyxpcm/issues/5), [#13](https://github.com/obidam/pyxpcm/issues/13), [#17](https://github.com/obidam/pyxpcm/issues/17), [#26](https://github.com/obidam/pyxpcm/issues/26)
- [x] Move from Travis to GHA for CI tests [#40](https://github.com/obidam/pyxpcm/pull/40)
