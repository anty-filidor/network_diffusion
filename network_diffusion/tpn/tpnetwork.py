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

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.utils import read_tpn


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
        mln_snaps = [MultilayerNetwork({"layer_1": net}) for net in snaps]
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
                MultilayerNetwork({"layer_0": snap})
                for _, snap in sorted(zip(snap_ids, network_list))
            ]
        else:
            snaps = [
                MultilayerNetwork({"layer_0": snap}) for snap in network_list
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
