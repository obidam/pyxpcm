# -*coding: UTF-8 -*-
#
# scipy==1.1.0
# xarray==0.10.7
# matplotlib==2.1.1
# scikit-learn==0.19.2

__author__ = 'gmaze@ifremer.fr'

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pyxpcm",
    version="0.3.1",
    description='pyxpcm: Ocean Profile Classification Model',
    url='http://github.com/obidam/pyxpcm',
    author='Guillaume Maze',
    author_email='gmaze@ifremer.fr',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={'pyxpcm': ['data/*.nc']},
    install_requires=requirements,
    license='GPLv3',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)