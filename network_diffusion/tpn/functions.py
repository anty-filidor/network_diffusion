
from typing import List, Set

import networkx as nx
from networkx.algorithms.approximation.dominating_set import min_weighted_dominating_set
from networkx.algorithms.dominating import dominating_set

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.tpn.tpnetwork import TemporalNetwork

def driver_nodes(net: MultilayerNetwork) -> List[MLNetworkActor]:
    """Return driver nodes for a given network"""

    #TODO: multi-layer dominating set need, now only handle the first layer
    # layer = net.layers['layer_1']   
    # min_dominating_set = dominating_set(layer)
    # min_dominating_set = min_weighted_dominating_set(net)

    min_dominating_set = []
    for layer in net.layers.values():
        min_dominating_set= minimum_dominating_set_with_initial(layer, min_dominating_set)

    return [net.get_actor(actor_id) for actor_id in min_dominating_set]


def minimum_dominating_set_with_initial(net: nx.Graph, initial_set: Set[int]) -> Set[int]:
    """
    Return a dominating set that includes the initial set.
    G: nx.Graph
    initial_set: set of nodes
    """
    if not set(initial_set).issubset(set(net.nodes())):
        raise ValueError("Initial set must be a subset of G's nodes")

    dominated = set(initial_set)
    dominating_set = set(initial_set)

    while len(dominated) < len(net):
        # Choose a node which dominates the maximum number of undominated nodes
        u = max(net.nodes(), key=lambda x: len(set(net[x]) - dominated))
        dominating_set.add(u)
        dominated.add(u)
        dominated.update(net[u])

    return dominating_set

