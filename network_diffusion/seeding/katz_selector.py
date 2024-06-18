# Copyright (c) 2023 by Damian Dąbrowski, Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition of the seed selector based on Katz centrality."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import katz
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class KatzSelector(BaseSeedSelector):
    """Katz Centrality seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tkatz centrality choice\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Katz centrality metric."""
        ranking_list: List[MLNetworkActor] = []
        # sorted by value of dictonaries
        for dc_val, _ in sorted(
            katz(net=net).items(), reverse=True, key=lambda x: x[1]
        ):
            # Adding actor to list
            ranking_list.append(dc_val)
        return ranking_list
