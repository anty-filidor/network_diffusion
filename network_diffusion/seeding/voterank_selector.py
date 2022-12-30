"""A definition of the seed selector based on Vote Rank algorithm."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.mln_actor import MLNetworkActor
from network_diffusion.mln.mln_network import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class VoteRankSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on Vote Rank algorithm."""

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes with Vote Rank algorithm.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        elected_nodes = nx.voterank(graph)
        unelected_nodes = set(graph.nodes).difference(set(elected_nodes))
        elected_nodes.extend(unelected_nodes)
        return elected_nodes

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tVoterank\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Compute ranking for actors."""
        return node_to_actor_ranking(super().nodewise(net), net)
