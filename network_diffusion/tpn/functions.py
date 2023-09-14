
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
    for layer in net.layers:
        min_dominating_set= minimum_dominating_set_with_initial(net, layer, min_dominating_set)

    return [net.get_actor(actor_id) for actor_id in min_dominating_set]


def minimum_dominating_set_with_initial(net: MultilayerNetwork, layer: str, initial_set: Set[int]) -> Set[int]:
    """
    Return a dominating set that includes the initial set.
    net: MultilayerNetwork
    layer: layer name
    initial_set: set of nodes
    """
    actor_ids = [x.actor_id for x in net.get_actors()]
    if not set(initial_set).issubset(set(actor_ids)):
        raise ValueError("Initial set must be a subset of net's actors")

    dominating_set = set(initial_set) 
    
    net_layer = net.layers[layer]
    isolated = set(actor_ids) - set(net_layer.nodes())
    dominating_set = dominating_set | isolated
    dominated = dominating_set

    while len(dominated) < len(net):
        # Choose a node which dominates the maximum number of undominated nodes
        u = max(net_layer.nodes(), key=lambda x: len(set(net_layer[x]) - dominated))
        dominating_set.add(u)
        dominated.add(u)
        dominated.update(net_layer[u])

    return dominating_set

