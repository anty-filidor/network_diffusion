# Copyright (c) 2022 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Functions for the auxiliary operations."""

import math
import pathlib
import random

import dynetx as dn
import networkx as nx
import numpy as np

BOLD_UNDERLINE = "============================================"
THIN_UNDERLINE = "--------------------------------------------"


def get_nx_snapshot(
    graph: dn.DynGraph | dn.DynDiGraph,
    snap_id: int,
    min_timestamp: int,
    time_window: int,
) -> nx.Graph | nx.DiGraph:
    """
    Get an `nx.Graph` typed snapshot for the given snapshot id.

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
) -> dict[int, nx.Graph | nx.DiGraph]:
    """
    Read temporal network from a text file for the TemporalNetwork class.

    :param file_path: path to file
    :return: a dictionary keyed by snapshot ID and valued by NetworkX Graph
    """
    net_dict = {}

    graph: dn.DynGraph | dn.DynDiGraph = dn.read_snapshots(
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


NumericType = int | float | np.number
