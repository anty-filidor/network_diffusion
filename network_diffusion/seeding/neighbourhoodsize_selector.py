"""A definition of the seed selector based on neighbourhood size."""

from typing import Any, Dict, List

import networkx as nx

from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE
from network_diffusion.mln.mln_network import MultilayerNetwork
from network_diffusion.mln.mln_actor import MLNetworkActor

class NeighbourhoodSizeSelector(BaseSeedSelector):
    """Neighbourhood Size seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tneighbourhood size choice\n{BOLD_UNDERLINE}\n"
        )

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )
    
    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Neighbourhood Size metric."""
        neighbourhood_size_values: Dict[int, List[MLNetworkActor]] = {}
        ranking_list: List[MLNetworkActor] = []

        for actor in net.get_actors():
            a_neighbours = set()
            for l_name in actor.layers:
                a_neighbours = a_neighbours.union(
                    set(net.layers[l_name].adj[actor.actor_id].keys())
                )
            a_neighbourhood_size = len(a_neighbours)
            if neighbourhood_size_values.get(a_neighbourhood_size) is None:
                neighbourhood_size_values[a_neighbourhood_size] = []
            neighbourhood_size_values[a_neighbourhood_size].append(actor)
        
        for ns_val in sorted(neighbourhood_size_values, reverse=True):
            ranking_list.extend(neighbourhood_size_values[ns_val])

        return ranking_list
