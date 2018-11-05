# -*coding: UTF-8 -*-
__author__ = 'gmaze@ifremer.fr'

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyxpcm",
    version="0.1.1",
    description='Profile Classification Model',
    url='http://github.com/obidam/pyxpcm',
    author='Guillaume Maze',
    author_email='gmaze@ifremer.fr',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)