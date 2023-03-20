#!/usr/bin/env python3

# Copyright 2022 by Michał Czuba, Piotr Bródka. All Rights Reserved.
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

from typing import List

from setuptools import find_packages, setup

from network_diffusion import __version__


def parse_requirements() -> List[str]:
    """Parse requirements from the txt file."""
    with open(
        file="requirements/production.txt", encoding="utf-8", mode="r"
    ) as file:
        requirements = file.readlines()
    return requirements


setup(
    name="network_diffusion",
    version=__version__,
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
        "Programming Language :: Python :: 3.10",
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
    description="Package to design and run diffusion phenomena in networks.",
    long_description="Network Diffusion",
    author="Michał Czuba, Piotr Bródka",
    author_email="michal.czuba@pwr.edu.pl, piotr.brodka@pwr.edu.pl",
    packages=find_packages(exclude=["*tests*"]),
    install_requires=parse_requirements(),
    python_requires=">=3.7",
)
