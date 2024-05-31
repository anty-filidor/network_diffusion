# Copyright 2023 by Michał Czuba, Piotr Bródka. All Rights Reserved.
#
# This file is part of Network Diffusion.
#
# Network Diffusion is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Network Diffusion is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# Network Diffusion. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""A definition of the seed selector based on neighbourhood size."""

from typing import Any, Dict, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.centrality_discount import (
    neighbourhood_size_discount,
)
from network_diffusion.mln.functions import neighbourhood_size
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

    def _calculate_ranking_list(self, graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError("Nodewise ranking list cannot be computed!")

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Neighbourhood Size metric."""
        neighbourhood_size_values: Dict[int, List[MLNetworkActor]] = {}
        ranking_list: List[MLNetworkActor] = []

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

    def _calculate_ranking_list(self, graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError("Nodewise ranking list cannot be computed!")

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Degree Centrality Discount algo."""
        return neighbourhood_size_discount(net=net, k=len(net))
