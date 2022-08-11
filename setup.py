#!/usr/bin/env python3

# Copyright 2020 by Michał Czuba, Piotr Bródka. All Rights Reserved.
#
# This file is part of Network Diffusion.
#
# Network Diffusion is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Network Diffusion is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# Network Diffusion. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Setup script to produce a package from the code."""

from setuptools import setup

setup(
    name="network_diffusion",
    version="0.7",
    url="https://github.com/anty-filidor/network_diffusion",
    project_urls={
        "Documentation": "https://network-diffusion.readthedocs.io/en/latest/",
        "Code": "https://github.com/anty-filidor/network_diffusion",
    },
    license="GPL",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 3.7",
    ],
    keywords=[
        "disease",
        "simulation",
        "processes influence",
        "network science",
        "multilayer networks",
        "networkx",
        "spreading",
        "phenomena",
    ],
    description="Package to design and run diffusion phenomena in networks",
    author="Michał Czuba, Piotr Bródka",
    author_email="michal.czuba.1995@gmail.com, piotr.brodka@pwr.edu.pl",
    packages=["network_diffusion"],
    install_requires=[
        "certifi>=2019.11.28",
        "cycler>=0.10.0",
        "decorator>=4.4.1",
        "kiwisolver>=1.1.0",
        "imageio>=2.6.1",
        "matplotlib>=3.1.1",
        "multipledispatch>=0.6.0",
        "networkx>=2.4",
        "numpy>=1.17.4",
        "olefile>=0.46",
        "pandas>=0.25.3",
        "Pillow>=6.2.1",
        "pyparsing>=2.4.5",
        "pytz>=2019.3",
        "six>=1.13.0",
        "tornado>=6.0.3",
        "tqdm>=4.40.2",
    ],
    python_requires=">=3.7",
)
