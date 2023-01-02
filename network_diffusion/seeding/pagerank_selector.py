"""A definition of the seed selector based on Page Rank algorithm."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.mln_actor import MLNetworkActor
from network_diffusion.mln.mln_network import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class PageRankSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on Page Rank algorithm."""

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes with Page Rank algorithm.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ranking_dict = nx.pagerank(graph)
        ranked_nodes = sorted(
            ranking_dict, key=lambda x: ranking_dict[x], reverse=True
        )
        if len(ranked_nodes) != len(graph.nodes):  # that's a sanity check
            assert ValueError
        return ranked_nodes

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\nPage Rank\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Compute ranking for actors."""
        return node_to_actor_ranking(super().nodewise(net), net)
