"""A definition of the seed selector based on neighbourhood size."""

from typing import Any, Dict, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
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

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
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
