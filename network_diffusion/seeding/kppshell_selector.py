# Copyright (c) 2024 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""K++ Shell based seed selector."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.kppshell import kppshell_seed_ranking
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class KPPShellSeedSelector(BaseSeedSelector):
    """Seed selector based on K++ Shell Decomposition algorithm."""

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """
        Create a ranking of nodes with K++ Shell Decomposition algorithm.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        return kppshell_seed_ranking(G=graph)

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tK++ Shell Decomposition \n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Compute ranking for actors."""
        return node_to_actor_ranking(super().nodewise(net), net)
