
"""A definition of the seed selector based on betweenness centrality."""

from typing import Any, Dict, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import betweennes
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class BetweennessSelector(BaseSeedSelector):
    """Betweenness Centrality seed selector."""

    def __str__(self) -> str:
        """Return seed method's description"""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tbetweenness centrality choice\n{BOLD_UNDERLINE}\n"
        )

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Betweenness Centrality metric."""
        betweennes_centrality_values: Dict[MLNetworkActor, float] = {}
        ranking_list: List[MLNetworkActor] = []

        for actor, a_betweennes in betweennes(net=net).items():
            #a betweennes's value is added to dict
            betweennes_centrality_values[actor] = a_betweennes
            #sorted by value of dictonaries
        for dc_val, val in sorted(betweennes_centrality_values.items(), reverse=True, key=lambda x: x[1]):
            #Adding actor to list
            ranking_list.append(dc_val)

        return ranking_list
