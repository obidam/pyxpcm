pyXpcm: Ocean Profile Classification Model
==========================================
[![DOI](https://img.shields.io/badge/DOI--Article-10.1016%2Fj.pocean.2016.12.008-orange.svg)](http://dx.doi.org/10.1016/j.pocean.2016.12.008)
[![Documentation Status](https://readthedocs.org/projects/pyxpcm/badge/?version=latest)](https://pyxpcm.readthedocs.io/en/latest/?badge=latest) 
[![Build Status](https://travis-ci.org/obidam/pyxpcm.svg?branch=master)](https://travis-ci.org/obidam/pyxpcm)  
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-270/)
[![](https://img.shields.io/badge/xarray-0.10.0-blue.svg)](http://xarray.pydata.org/en/stable/)  
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) 

**pyXpcm** is a python package to create and work with ocean **Profile Classification Model** that consumes and produces [Xarray](https://github.com/pydata/xarray) objects. Xarray objects are N-D labeled arrays and datasets in Python. 

A ocean **Profile Classification Model** allows to automatically assemble ocean profiles in clusters according to their vertical structure similarities.   
The geospatial properties of these clusters can be used to address a large variety of oceanographic problems: front detection, water mass identification, natural region contouring (gyres, eddies), reference profile selection for QC validation, etc... The vertical structure of these clusters furthermore provides a highly synthetic representation of large ocean areas that can be used for dimensionality reduction and coherent intercomparisons of ocean data (re)-analysis or simulations.   

## Why pyXpcm?
The **Ocean dynamics** and its 3-dimensional structure and variability is so complex that it is very difficult to develop objective and efficient diagnostics of horizontally and vertically coherent oceanic patterns. However, identifying such **patterns** is crucial to the understanding of interior mechanisms as, for instance, the integrand giving rise to Global Ocean Indicators (e.g. heat content and sea level rise). We believe that, by using state of the art **machine learning** algorithms and by building on the increasing availability of ever-larger **in situ and numerical model datasets**, we can address this challenge in a way that was simply not possible a few years ago. Following this approach, **Profile Classification Modelling** focuses on the smart identification of vertically coherent patterns and their space/time distribution and occurrence.

## Documentation
[https://pyxpcm.readthedocs.io](https://pyxpcm.readthedocs.io)

## Install

Latest release:

    pip install pyxpcm
    
Development version:

    pip install git+https://github.com/obidam/pyxpcm.git

    