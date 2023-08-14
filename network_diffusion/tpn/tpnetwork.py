"""A script where a temporal network is defined."""

from typing import Any, Dict, List, Optional

import networkx as nx

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
        snap_ids: Optional[List[Any]] = None,
    ) -> "TemporalNetwork":
        """
        Load a temporal network from a list of networks and their snapshot ids.

        :param network_list: a list of nx networks
        :param snap_ids:  list of snapshot ids. It can be none, then ids
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
        return cls(snaps)
