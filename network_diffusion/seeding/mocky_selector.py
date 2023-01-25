"""A definition of "selector" that returns aprioiry provided actors."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class MockyActorSelector(BaseSeedSelector):
    """Mocky seed selector - returns a ranking provided as argument to init."""

    def __init__(self, preselected_actors: List[MLNetworkActor]) -> None:
        """Initialise object."""
        self.preselected_actors = preselected_actors

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tmocked choice - seeds provided a priori\n{BOLD_UNDERLINE}\n"
        )

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors."""
        if net.get_actors_num() != len(self.preselected_actors):
            raise ValueError("Incorrect length of preselected actors!")
        # we can also check if they are from the network, but it's too padantic
        return self.preselected_actors
