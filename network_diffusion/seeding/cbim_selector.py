# Copyright (c) 2025 by Damian Dąbrowski, Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition community based influence maximization selector class."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.cbim import cbim_seed_ranking
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class CBIMSeedselector(BaseSeedSelector):
    """
    CBIM seed selector.

    The this implementation is based on algorithm presented by Venkatakrishna
    Rao, C. Ravindranath Chowdary in CBIM: Community-based influence
    maximization in multilayer networks" which was published in "Information
    Sciences", 2022, Volume 609 (https://doi.org/10.1016/j.ins.2022.07.103).
    Call "nodewise" method for the original implementation (which ranks
    separately nodes from each later). To select actors use "actorwise" method.
    """

    def __init__(self, merging_idx_threshold: float) -> None:
        """
        Create an object.

        :param merging_idx_threshold: a threshold above which communities will
            not be merged anymore during consolidation phase
        """
        assert 0 < merging_idx_threshold <= 1, "par. must be in range [0, 1]!"
        super().__init__()
        self.merging_idx_threshold = merging_idx_threshold  # 0.1

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tCBIM choice\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        return cbim_seed_ranking(
            net=graph,
            merging_idx_threshold=self.merging_idx_threshold,
            debug=False,
            weight_attr=None,
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking using Community Based Influence Maximization."""
        return node_to_actor_ranking(super().nodewise(net), net)
