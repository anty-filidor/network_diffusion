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

"""A definition of the seed selectors based on Vote Rank algorithm."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import voterank_actorwise
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class VoteRankSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on Vote Rank algorithm."""

    def _calculate_ranking_list(self, graph: nx.Graph) -> List[Any]:
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


class VoteRankMLNSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on Vote Rank algorithm."""

    def _calculate_ranking_list(self, graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tVoterank computed actorwise\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Compute ranking for actors."""
        elected_nodes = voterank_actorwise(net=net)
        unelected_nodes = set(net.get_actors()).difference(set(elected_nodes))
        elected_nodes.extend(unelected_nodes)
        return elected_nodes
