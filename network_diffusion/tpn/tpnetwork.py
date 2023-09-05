"""A script where a temporal network is defined."""

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.utils import read_tpn


# TODO: relax assumption that snaps are ordinary numbers
class TemporalNetwork:
    """Container for a temporal network."""

    def __init__(self, snaps: Dict[int, MultilayerNetwork]) -> None:
        """
        Create a temporal network object.

        :param snaps: snapshots of the temporal network.
        """
        self.snaps = snaps  # TODO: change it to list!

    def __getitem__(self, key: int) -> nx.Graph:
        """Get 'key' snapshot of the network."""
        return self.snaps[key]

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
        mln_snaps = {
            idx: MultilayerNetwork({"layer_1": net})
            for idx, net in enumerate(snaps)
        }
        return cls(mln_snaps)

    @classmethod
    def from_nx_layers(
        cls, network_list: List[nx.Graph], snap_ids: Optional[List[Any]] = None
    ) -> "TemporalNetwork":
        """
        Load a temporal network from a list of networks and their snapshot ids.

        :param network_list: a list of nx networks
        :param snap_ids: list of snapshot ids. It can be none, then ids
            are set automatically
        """
        snaps = {}
        if snap_ids is not None:
            assert len(network_list) == len(
                snap_ids
            ), "Length of network list and metadata list is not equal"
            for snap, snap_id in zip(network_list, snap_ids):
                snaps.update({snap_id: MultilayerNetwork({"layer_0": snap})})
        else:
            for snap_id, snap in enumerate(network_list, start=0):
                snaps.update({snap_id: MultilayerNetwork({"layer_0": snap})})
        return cls(snaps)

    def __len__(self) -> int:
        """Return length of the network, i.e. num of actors."""
        return len(self.snaps)

    def get_actors(self, shuffle: bool = False) -> List[MLNetworkActor]:
        """
        Get actors that from the first snapshot of network and their states.

        :param shuffle: a flag that determines whether to shuffle actor list
        :return: a list with actors that live in the network
        """
        return self.get_actors_from_snap(0, shuffle)

    def get_actors_from_snap(
        self, snapshot_id: int, shuffle: bool = False
    ) -> List[MLNetworkActor]:
        """
        Get actors that exist in the network at given snapshotand read their states.

        :param snapshot_id: snapshot for which to take actors
        :param shuffle: a flag that determines whether to shuffle actor list
        :return: a list with actors that live in the network
        """
        return list(self.snaps[snapshot_id].get_actors(shuffle))

    def get_actors_num(self) -> int:
        """Get number of actors that live in the network."""
        return len(self.get_actors_from_snap(0))

    # def get_states_num(
    #     self, snapshot_id: str
    # ) -> Dict[str, Tuple[Tuple[Any, int], ...]]:
    #     """
    #     Return number of agents with all possible states in each layer.

    #     :return: dictionary with items representing summary of nodes states in values
    #     """
    #     statistics = {}
    #     tab = []
    #     graph = self.snaps[snapshot_id]["layer_0"]
    #     for node in graph.nodes():
    #         # TODO: problem with visualization when encoding all agent features under the status field, it makes each status unique because of the belief and evidence values - we'd like to operate on actors'/nodes' states, not statuses
    #         # tab.append(graph.nodes[node]["status"])
    #         tab.append(graph.nodes[node]["status"].split("_")[0])
    #     statistics["layer_0"] = tuple(Counter(tab).items())
    #     return statistics
