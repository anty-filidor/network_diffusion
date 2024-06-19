# Copyright (c) 2023 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition of the seed selector based on betweenness centrality."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import betweenness
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class BetweennessSelector(BaseSeedSelector):
    """Betweenness Centrality seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tbetweenness centrality choice\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking for actors using betweenness centrality metric."""
        ranking_list: list[MLNetworkActor] = []
        # sorted by value of dictonaries
        for dc_val, _ in sorted(
            betweenness(net=net).items(), reverse=True, key=lambda x: x[1]
        ):
            # Adding actor to list
            ranking_list.append(dc_val)
        return ranking_list
