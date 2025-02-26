# Copyright (c) 2023 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition of the seed selector based on degree centrality."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.centralities import degree
from network_diffusion.mln.centrality_discount import (
    degree_discount,
    degree_discount_networkx,
)
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class DegreeCentralitySelector(BaseSeedSelector):
    """Degree Centrality seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tdegree centrality choice\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list is not implemented for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking for actors using Degree Centrality metric."""
        degree_centrality_values: dict[int, list[MLNetworkActor]] = {}
        ranking_list: list[MLNetworkActor] = []

        for actor, a_degree in degree(net=net).items():
            if degree_centrality_values.get(a_degree) is None:
                degree_centrality_values[a_degree] = []
            degree_centrality_values[a_degree].append(actor)

        for dc_val in sorted(degree_centrality_values, reverse=True):
            ranking_list.extend(degree_centrality_values[dc_val])

        return ranking_list


class DegreeCentralityDiscountSelector(BaseSeedSelector):
    """Degree Centrality Discount seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tDegree Centrality Discount\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking."""
        return degree_discount_networkx(net=graph, k=len(graph.nodes))

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking for actors using Degree Centrality Discount algo."""
        return degree_discount(net=net, k=len(net))
