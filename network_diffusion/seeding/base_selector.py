"""A definition of the base seed selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


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
    Nodewise / actorwise converter of seed ranking by weighted average.

    :param nodewise_ranking: a ranking computed with BaseSeedSelector.nodewise
    :param net: a network to convert ranking for
    :return: ranking of actors according to provided nodes' ranking
    """
    l_sizes = {l_name: len(l_graph) for l_name, l_graph in net.layers.items()}

    # obtain score of each actor in its layers
    actor_partial_scores: Dict[str, Dict[str, int]] = {}
    for l_name, l_ranking in nodewise_ranking.items():
        for idx, node_id in enumerate(l_ranking, start=1):
            if node_id not in actor_partial_scores:
                actor_partial_scores[node_id] = {}
            actor_partial_scores[node_id][l_name] = idx

    def _avg_result(results: Dict[str, int], sizes: Dict[str, int]) -> float:
        """Compute average score in ranking weighted by size of layer."""
        counter = sum(
            [l_result * sizes[l_name] for l_name, l_result in results.items()]
        )
        denominator = sum(l_sizes.values())
        return counter / denominator

    # compute average score for each actor and create an ordered list of them
    actor_ranking = sorted(
        _ := {
            net.get_actor(actor_id): _avg_result(partial_score, l_sizes)
            for actor_id, partial_score in actor_partial_scores.items()
        },
        key=_.get,  # type: ignore
    )

    assert len(actor_ranking) == net.get_actors_num(), "Incorrect ranking!"
    return actor_ranking
