# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition of the seed selector based on neighbourhood size."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.centralities import neighbourhood_size
from network_diffusion.mln.centrality_discount import (
    neighbourhood_size_discount,
)
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class NeighbourhoodSizeSelector(BaseSeedSelector):
    """Neighbourhood Size seed selector."""

    def __init__(self, connection_hop: int = 1) -> None:
        """Initialise object."""
        super().__init__()
        self.connection_hop = connection_hop

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tneighbourhood size choice with hop size: {self.connection_hop}"
            f"\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError("Nodewise ranking list cannot be computed!")

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking for actors using Neighbourhood Size metric."""
        neighbourhood_size_values: dict[int, list[MLNetworkActor]] = {}
        ranking_list: list[MLNetworkActor] = []

        for actor, a_nsize in neighbourhood_size(
            net=net, connection_hop=self.connection_hop
        ).items():
            if neighbourhood_size_values.get(a_nsize) is None:
                neighbourhood_size_values[a_nsize] = []
            neighbourhood_size_values[a_nsize].append(actor)

        for ns_val in sorted(neighbourhood_size_values, reverse=True):
            ranking_list.extend(neighbourhood_size_values[ns_val])

        return ranking_list


class NeighbourhoodSizeDiscountSelector(BaseSeedSelector):
    """Neighbourhood Size Discount seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tNeighbourhood Size Discount\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError("Nodewise ranking list cannot be computed!")

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking for actors using Degree Centrality Discount algo."""
        return neighbourhood_size_discount(net=net, k=len(net))
