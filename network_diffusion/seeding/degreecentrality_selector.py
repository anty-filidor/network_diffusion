"""A definition of the seed selector based on degree centrality."""

from typing import Any, Dict, List

import networkx as nx

from network_diffusion.mln.mln_actor import MLNetworkActor
from network_diffusion.mln.mln_network import MultilayerNetwork
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

        for actor in net.get_actors():
            a_neighbours = 0
            for l_name in actor.layers:
                a_neighbours += len(net.layers[l_name].adj[actor.actor_id])
            if degree_centrality_values.get(a_neighbours) is None:
                degree_centrality_values[a_neighbours] = []
            degree_centrality_values[a_neighbours].append(actor)

        for dc_val in sorted(degree_centrality_values, reverse=True):
            ranking_list.extend(degree_centrality_values[dc_val])

        return ranking_list
