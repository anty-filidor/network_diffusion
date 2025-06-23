# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Functions for the auxiliary operations on temporal networks."""

import math

import dynetx as dn
import networkx as nx


def _get_nx_snapshot(
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
    Read temporal network from a text file for the dict of NetworkX networks.

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
        net_dict[snap_id] = _get_nx_snapshot(
            graph, snap_id, min_timestamp, time_window
        )

    return net_dict
