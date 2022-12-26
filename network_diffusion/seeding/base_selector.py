"""A definition of the base seed selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import networkx as nx

from network_diffusion.multilayer_network import MultilayerNetwork


class BaseSeedSelector(ABC):
    """Base abstract class for seed selectors."""

    def __call__(self, network: MultilayerNetwork) -> Dict[str, List[Any]]:
        """
        Prepare ranking list according to implementing seeding strategy.

        :param network: a network to calculate ranking list for
        :return: a dictionary keyed by layer names with lists of node names
            ordered in descending way according to seed selection strategy
        """
        nodes_ranking = {}
        for l_name, l_graph in network.layers.items():
            seeds_in_layer = self._calculate_ranking_list(l_graph)
            nodes_ranking[l_name] = seeds_in_layer
        return nodes_ranking

    @staticmethod
    @abstractmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes based on concrete metric/heuristic.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ...
