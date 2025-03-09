# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition of the seed selectors based on Page Rank algorithm."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import squeeze_by_neighbourhood
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class PageRankSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on Page Rank algorithm."""

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
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
            raise ValueError
        return ranked_nodes

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\nPage Rank\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Compute ranking for actors."""
        return node_to_actor_ranking(super().nodewise(net), net)


class PageRankMLNSeedSelector(PageRankSeedSelector):
    """Selector for MLTModel based on Page Rank algorithm."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\nPage Rank computed actorwise\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Compute ranking for actors."""
        squeezed_net = squeeze_by_neighbourhood(net)
        return self._calculate_ranking_list(squeezed_net)
