"""A script where a temporal network is defined."""

import networkx as nx
import random

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from network_diffusion.utils import read_tpn


class TemporalNetwork:
    """Container for a temporal network."""

    def __init__(self, snaps: Dict[int, nx.Graph]) -> None:
        """
        Create a temporal network object.

        :param snaps: snapshots of the temporal network.
        """
        self.snaps = snaps

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
        return cls(snaps)

    @classmethod
    def from_nx_layers(
        cls,
        network_list: List[nx.Graph],
        snap_ids: Optional[List[Any]] = None
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
                snaps.update({snap_id: snap})
        else:
            for snap_id, snap in enumerate(network_list, start=0):
                snaps.update({snap_id: snap})
        prepared_snaps = cls._prepare_nodes_attribute(snaps)
        return cls(prepared_snaps)

    @staticmethod
    def _prepare_nodes_attribute(snaps: Dict[str, nx.Graph]) -> Dict[str, nx.Graph]:
        """Prepare network to the experiment."""
        for snap in snaps.values():
            status_dict = {n: None for n in snap.nodes()}
            nx.set_node_attributes(snap, status_dict, "status")
        return snaps

    def get_size(self):
        return len(self.snaps)

    def get_actors(self, shuffle: bool = False) -> List[Dict[int, Dict]]:
        """
        Get actors that exist in the network and read their states.

        :param shuffle: a flag that determines whether to shuffle actor list
        :return: a list with actors that live in the network
        """
        agent_list = list(self.snaps[0].nodes(data=True))
        if shuffle:
            random.shuffle(agent_list)
        return agent_list

    def get_actors_from_snap(self, snapshot_id: int, shuffle: bool = False) -> List[Dict[int, Dict]]:
        """
        Get actors that exist in the network at given snapshotand read their states.

        :param snapshot_id: snapshot for which to take actors
        :param shuffle: a flag that determines whether to shuffle actor list
        :return: a list with actors that live in the network
        """
        agent_list = list(self.snaps[snapshot_id].nodes(data=True))
        if shuffle:
            random.shuffle(agent_list)
        return agent_list

    def get_actors_num(self) -> int:
        """Get number of actors that live in the network."""
        actors_set: Set[Any] = set(self.snaps[list(self.snaps.keys())[0]].nodes)
        return len(actors_set)

    def get_states_num(self, snapshot_id: str) -> Dict[str, Tuple[Tuple[Any, int], ...]]:
        """
        Return number of agents with all possible states in each layer.

        :return: dictionary with items representing summary of nodes states in values
        """
        statistics = {}
        tab = []
        graph = self.snaps[snapshot_id]
        for node in graph.nodes():
            # TODO: problem with visualization when encoding all agent features under the status field, it makes each status unique because of the belief and evidence values - we'd like to operate on actors'/nodes' states, not statuses
            #tab.append(graph.nodes[node]["status"])
            tab.append(graph.nodes[node]["status"].split('_')[0])
        statistics["TPN"] = tuple(Counter(tab).items())
        return statistics
