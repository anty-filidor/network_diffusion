"""A definition of the base seed selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import networkx as nx

from network_diffusion.mln.mln_network import MultilayerNetwork
from network_diffusion.mln.mln_actor import MLNetworkActor


class BaseSeedSelector(ABC):
    """Base abstract class for seed selectors."""

    def __call__(
        self, network: MultilayerNetwork, actorwise: bool = False
    ) -> Union[Dict[str, List[Any]], List[MLNetworkActor]]:
        """
        Prepare ranking list according to implementing seeding strategy.

        :param network: a network to calculate ranking list for
        :param actorwise: a flag wether to return ranking by actors, instead of
            computing it for nodes (which is a default way)
        :return: a dictionary keyed by layer names with lists of node names
            (or just a list of actors) ordered in descending way according to
            seed selection strategy
        """
        if actorwise:
            return self.actorwise(net=network)
        else:
            return self.nodewise(net=network)
    
    @abstractmethod
    def __str__(self) -> str:
        """Return seed method's description."""
        ...

    @staticmethod
    @abstractmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes based on concrete metric/heuristic.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ...

    @abstractmethod
    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Create actorwise ranking."""
        ...
    
    def nodewise(self, net: MultilayerNetwork) -> Dict[str, List[Any]]:
        """Create nodewise ranking."""
        nodes_ranking = {}
        for l_name, l_graph in net.layers.items():
            seeds_in_layer = self._calculate_ranking_list(l_graph)
            nodes_ranking[l_name] = seeds_in_layer
        return nodes_ranking
