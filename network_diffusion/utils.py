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

"""Functions for the auxiliary operations."""

import math
import pathlib
import random
import string
from typing import Any, Dict, List, Union

import dynetx as dn
import networkx as nx
import numpy as np

BOLD_UNDERLINE = "============================================"
THIN_UNDERLINE = "--------------------------------------------"


def read_mpx(file_path: str) -> Dict[str, List[Any]]:
    """
    Handle MPX file for the MultilayerNetwork class.

    :param file_path: path to file

    :return: a dictionary with network to create class
    """
    # initialise empty containers
    net_dict = {}
    tab: List[Any] = []
    name = "foo"

    with open(file=file_path, mode="r", encoding="utf-8") as file:
        line = file.readline()

        # omit trash
        while line and line[0] != "#":
            line = file.readline()

        # read pure data
        while line:
            # if line contains title of new division
            if line[0] == "#":
                # if this is a special line - type
                if "#TYPE".lower() in line.lower():
                    net_dict.update(
                        {"type": [line[6:-1]]}
                    )  # '6:-1' to save the name of type
                # else if it is a normal division
                else:
                    net_dict.update({name: tab})
                    name = line[1:-1].lower()  # omitting '#' and '\n'
                    tab = []
            line = file.readline()
            # don't save line with only whitespaces or if line contains a
            # title of new division
            if not line.isspace() and "#" not in line:
                line = line.translate(
                    {ord(char): None for char in string.whitespace}
                )
                tab.append(line.split(","))

        # append last line to dictionary
        net_dict.update({name: tab[:-1]})
    del net_dict["foo"]

    return net_dict


def get_nx_snapshot(
    graph: Union[dn.DynGraph, dn.DynDiGraph],
    snap_id: int,
    min_timestamp: int,
    time_window: int,
) -> Union[nx.Graph, nx.DiGraph]:
    """
    Get an nxGraph typed snapshot for the given snapshot id.

    :param graph: the dynamic graph
    :param snap_id: the snapshot id
    :param min_timestamp: the minimum timestamp in the graph
    :param time_window: the size of the time window
    :return: a snapshot graph of the given id
    """
    win_begin = snap_id * time_window + min_timestamp
    win_end = (snap_id + 1) * time_window + min_timestamp
    if isinstance(graph, dn.DynGraph):
        return nx.Graph(graph.time_slice(t_from=win_begin, t_to=win_end).edges)
    elif isinstance(graph, dn.DynDiGraph):
        return nx.DiGraph(
            graph.time_slice(t_from=win_begin, t_to=win_end).edges
        )
    else:
        raise ValueError("Incorrect class of graph!")


def read_tpn(
    file_path: str, time_window: int, directed: bool = True
) -> Dict[int, Union[nx.Graph, nx.DiGraph]]:
    """
    Read temporal network from a text file for the TemporalNetwork class.

    :param file_path: path to file
    :return: a dictionary keyed by snapshot ID and valued by NetworkX Graph
    """
    net_dict = {}

    graph: Union[dn.DynGraph, dn.DynDiGraph] = dn.read_snapshots(
        file_path, directed=directed, nodetype=int, timestamptype=int
    )
    min_timestamp = min(graph.temporal_snapshots_ids())
    max_timestamp = max(graph.temporal_snapshots_ids())
    num_of_snaps = math.ceil((max_timestamp - min_timestamp) / time_window)

    for snap_id in range(num_of_snaps):
        net_dict[snap_id] = get_nx_snapshot(
            graph, snap_id, min_timestamp, time_window
        )

    return net_dict


def get_absolute_path() -> str:
    """Get absolute path of the library's sources."""
    return str(pathlib.Path(__file__).parent)


def fix_random_seed(seed: int) -> None:
    """Fix pseudo-random number generator seed for reproducible tests."""
    random.seed(seed)
    np.random.seed(seed)


NumericType = Union[int, float, np.number]
