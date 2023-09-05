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
# pylint: disable=R0913  # it's a scientific function, so ignore nb of args


from typing import List


def cogsnet(
    forgetting_type: str,
    snapshot_interval: int,
    edge_lifetime: int,
    mu: float,
    theta: float,
    units: int,
    path_events: str,
    delimiter: str,
) -> List[List[float]]:
    """
    Load a temporal network from a csv file with events and create CogSNet with the given parameters.

    Note, the csv file should be in the form of: SRC DST TIME.
    The timestamps TIME should be in ascending order.
    The timestamps are expected to be provided in seconds.

    :param str forgetting_type: The forgetting function used to decrease the weight of edges over time. Allowed values are 'exponential', 'power', or 'linear'.
    :param int snapshot_interval: The interval for taking snapshots (0 or larger) expressed in units param (seconds, minutes, hours). A value of 0 means taking a snapshot after each event.
    :param int edge_lifetime: The lifetime of an edge after which the edge will disappear if no new event occurs (greater than 0).
    :param float mu: The value of increasing the weight of an edge for each new event (greater than 0 and less than or equal to 1).
    :param float theta: The cutoff point (between 0 and mu). If the weight falls below theta, the edge will disappear.
    :param int units: The time units (1 for seconds, 60 for minutes, or 3600 for hours). For the power forgetting function, this parameter also determines the minimum interval between events to prevent them from being skipped when calculating the weight.
    :param str path_events: The path to the CSV file with events.
    :param str delimiter: The delimiter for the CSV file (allowed values are ',', ';', or '\\t').

    :return: List[List[float]] A list of snapshots, where each snapshot contains a list of edges with their weights (SRC, DST, WEIGHT).
    """
    ...
