"""A definition of the base seed selector class."""

from abc import ABC, abstractmethod
from itertools import zip_longest
from random import shuffle
from typing import Any, Dict, List, Union

import networkx as nx

from network_diffusion.mln.mln_actor import MLNetworkActor
from network_diffusion.mln.mln_network import MultilayerNetwork


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


def node_to_actor_ranking(
    nodewise_ranking: Dict[str, List[Any]], net: MultilayerNetwork
) -> List[MLNetworkActor]:
    """
    Naive converter of seeding ranking computed nodewise to actorwise.

    :param nodewise_ranking: a ranking computed with BaseSeedSelector.nodewise
    :param net: a network to convert ranking for
    :return: ranking of actors according to provided nodes' ranking
    """
    actor_ranking, picked_actors = [], []
    for nth_nodes in zip_longest(*nodewise_ranking.values(), fillvalue=None):
        shuffle(list(nth_nodes))
        for node_id in nth_nodes:
            if node_id is None or node_id in picked_actors:
                continue
            actor_ranking.append(net.get_actor(actor_id=node_id))
            picked_actors.append(node_id)
    assert len(picked_actors) == net.get_actors_num(), "Incorrect ranking!"
    return actor_ranking
