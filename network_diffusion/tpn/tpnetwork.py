#!/usr/bin/env python3

# Copyright 2023 by Yuxuan Qiu. All Rights Reserved.
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

"""A script where a temporal network is defined."""

from typing import Any, List, Optional

import networkx as nx
import pandas as pd

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.tpn.cogsnet_lib import _cogsnet  # pylint: disable=E0611
from network_diffusion.utils import read_tpn


def cogsnet_snap_to_nxgraph(cogsnet_snap: List[List[float]]) -> nx.DiGraph:
    """
    Create nx.DiGraph from one snapshot obtained from CogSNet.

    :param cogsnet_snap: a CogSNet snapshot
    """
    edges_df = pd.DataFrame(
        cogsnet_snap, columns=["source", "target", "weight"]
    )
    edges_df = edges_df.astype({"source": int, "target": int, "weight": float})
    return nx.from_pandas_edgelist(
        edges_df, create_using=nx.DiGraph, edge_attr="weight"
    )


class TemporalNetwork:
    """Container for a temporal network."""

    def __init__(self, snaps: List[MultilayerNetwork]) -> None:
        """
        Create a temporal network object.

        :param snaps: snapshots of the temporal network.
        """
        self.snaps = snaps

    def __getitem__(self, key: int) -> MultilayerNetwork:
        """Get 'key' snapshot of the network."""
        return self.snaps[key]

    def __len__(self) -> int:
        """Return length of the network, i.e. num of actors."""
        return len(self.snaps)

    @classmethod
    def from_txt(
        cls, file_path: str, time_window: int, directed: bool = True
    ) -> "TemporalNetwork":
        """
        Load a temporal network from a txt file.

        Note, the txt file should be in the form of: SRC DST UNIXTS.
        The timestamps UNIXTS should be in ascending order.
        The timestamps are expected to be provided in seconds.

        :param file_path: path to the file
        :param time_window: the time window size for each snapshot
        :param directed: indicate if the graph is directed
        """
        snaps = read_tpn(file_path, time_window, directed)
        mln_snaps = [
            MultilayerNetwork({"layer_1": net}) for net in snaps.values()
        ]
        return cls(mln_snaps)

    @classmethod
    def from_nx_layers(
        cls, network_list: List[nx.Graph], snap_ids: Optional[List[Any]] = None
    ) -> "TemporalNetwork":
        """
        Load a temporal network from a list of networks and their snapshot ids.

        :param network_list: a list of nx networks
        :param snap_ids: list of snapshot ids. It can be none, then ids
            are set automatically, if not, then snapshots will be sorted
            according to snap_ids list
        """
        if snap_ids is not None:
            assert len(network_list) == len(
                snap_ids
            ), "Length of network list and metadata list is not equal"
            snaps = [
                MultilayerNetwork({"layer_1": snap})
                for _, snap in sorted(zip(snap_ids, network_list))
            ]
        else:
            snaps = [
                MultilayerNetwork({"layer_1": snap}) for snap in network_list
            ]
        return cls(snaps)

    def get_actors(self, shuffle: bool = False) -> List[MLNetworkActor]:
        """
        Get actors that from the first snapshot of network.

        :param shuffle: a flag that determines whether to shuffle actor list
        """
        return self.get_actors_from_snap(0, shuffle)

    def get_actors_from_snap(
        self, snapshot_id: int, shuffle: bool = False
    ) -> List[MLNetworkActor]:
        """
        Get actors that exist in the network at given snapshot.

        :param snapshot_id: snapshot for which to take actors, starts from 0
        :param shuffle: a flag that determines whether to shuffle actor list
        """
        return list(self.snaps[snapshot_id].get_actors(shuffle))

    def get_actors_num(self) -> int:
        """Get number of actors that live in the network."""
        return len(self.get_actors_from_snap(0))

    @classmethod
    def from_cogsnet(  # pylint: disable=R0913, E1111, E1133
        cls,
        forgetting_type: str,
        snapshot_interval: int,
        edge_lifetime: int,
        mu: float,  # pylint: disable=C0103
        theta: float,
        units: int,
        path_events: str,
        delimiter: str,
    ) -> "TemporalNetwork":
        r"""
        Load events from a csv file and create CogSNet.

        Note, the csv file should be in the form of: SRC DST TIME.
        The timestamps TIME should be in ascending order.
        The timestamps are expected to be provided in seconds.

        :param str forgetting_type: The forgetting function used to decrease
            the weight of edges over time. Allowed values are 'exponential',
            'power', or 'linear'.
        :param int snapshot_interval: The interval for taking snapshots
            (0 or larger) expressed in units param (seconds, minutes, hours).
            A value of 0 means taking a snapshot after each event.
        :param int edge_lifetime: The lifetime of an edge after which the edge
            will disappear if no new event occurs (greater than 0).
        :param float mu: The value of increasing the weight of an edge for each
            new event (greater than 0 and less than or equal to 1).
        :param float theta: The cutoff point (between 0 and mu).
            If the weight falls below theta, the edge will disappear.
        :param int units: The time units (1 for seconds, 60 for minutes,
            or 3600 for hours). For the power forgetting function,
            this parameter also determines the minimum interval between events
            to prevent them from being skipped when calculating the weight.
        :param str path_events: The path to the CSV file with events.
        :param str delimiter: The delimiter for the CSV file
            (allowed values are ',', ';', or '\\t').
        """
        cogsnet_snaps = _cogsnet(
            forgetting_type,
            snapshot_interval,
            edge_lifetime,
            mu,  # pylint: disable=C0103
            theta,
            units,
            path_events,
            delimiter,
        )
        mln_snaps = [
            MultilayerNetwork({"layer_1": cogsnet_snap_to_nxgraph(snap)})
            for snap in cogsnet_snaps
        ]
        return cls(mln_snaps)  # TODO: add to docs!
