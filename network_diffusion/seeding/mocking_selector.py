# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A definition of "selector" that returns aprioiry provided actors."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class MockingActorSelector(BaseSeedSelector):
    """Mock seed selector - returns a ranking provided as argument to init."""

    def __init__(self, preselected_actors: list[MLNetworkActor]) -> None:
        """Initialise object."""
        self.preselected_actors = preselected_actors
        super().__init__()

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tmocked choice - seeds provided a priori\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Get ranking for actors."""
        if net.get_actors_num() != len(self.preselected_actors):
            raise ValueError("Incorrect length of preselected actors!")
        # we can also check if they are from the network, but it's too padantic
        return self.preselected_actors
