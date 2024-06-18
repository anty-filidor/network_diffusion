# Copyright (c) 2023 by Mateusz Nurek, Radosław Michalski, Michał Czuba.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
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
