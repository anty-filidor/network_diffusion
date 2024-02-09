#!/usr/bin/env python3

# Copyright 2023 by Mateusz Nurek. All Rights Reserved.
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

"""
Interface for C ports for CogSNet functions.

This file needs to have a .so lib installed in the scope of parent module. You
can build it with `python setup.py build`.
"""

# pylint: disable=W0613  # it's a stub for C functions
# pylint: disable=R0913, C0103  # it's a scientific function, so ignore SOLID
# mypy: disable-error-code=empty-body


from typing import List


def _cogsnet(
    forgetting_type: str,
    snapshot_interval: int,
    edge_lifetime: int,
    mu: float,
    theta: float,
    units: int,
    path_events: str,
    delimiter: str,
) -> List[List[List[float]]]:
    """Call cogsnet function in C and get snapshots of weighted edgelists."""
