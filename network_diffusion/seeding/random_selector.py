# Copyright (c) 2023 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Randomised seed selector."""

from random import shuffle
from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class RandomSeedSelector(BaseSeedSelector):
    """Randomised seed selector prepared mainly for DSAA algorithm."""

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """
        Create a random ranking of nodes.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        nodes_list = [*graph.nodes()]
        shuffle(nodes_list)
        return nodes_list

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tnodewise random choice\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get actors randomly."""
        return net.get_actors(shuffle=True)
