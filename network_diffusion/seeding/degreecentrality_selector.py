"""A definition of the seed selector based on degree centrality."""

from typing import Any, Dict, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import degree
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class DegreeCentralitySelector(BaseSeedSelector):
    """Degree Centrality seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tdegree centrality choice\n{BOLD_UNDERLINE}\n"
        )

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Degree Centrality metric."""
        degree_centrality_values: Dict[int, List[MLNetworkActor]] = {}
        ranking_list: List[MLNetworkActor] = []

        for actor, a_degree in degree(net=net).items():
            if degree_centrality_values.get(a_degree) is None:
                degree_centrality_values[a_degree] = []
            degree_centrality_values[a_degree].append(actor)

        for dc_val in sorted(degree_centrality_values, reverse=True):
            ranking_list.extend(degree_centrality_values[dc_val])

        return ranking_list
