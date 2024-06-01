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

from setuptools import Extension, find_packages, setup


def parse_requirements() -> List[str]:
    """Parse requirements from the txt file."""
    with open(file="requirements/production.txt", encoding="utf-8") as file:
        requirements = file.readlines()
    return requirements


def parse_readme() -> str:
    """Convert README to rst standard."""
    with open("README.md", encoding="utf-8") as file:
        long_description = file.read()
    return long_description


setup(
    long_description=parse_readme(),
    long_description_content_type="text/markdown",
    install_requires=parse_requirements(),
    ext_modules=[
        Extension(
            name="network_diffusion.tpn.cogsnet_lib",
            include_dirs=["c_modules"],
            sources=["c_modules/cogsnet_compute.c", "c_modules/cogsnet_lib.c"],
        )
    ],
    packages=find_packages(exclude=["*tests*"]),
)
