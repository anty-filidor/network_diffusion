# Copyright (c) 2025 by Yu-Xuan Qi, Mingshan Jia, Michał Czuba.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Script with functions for driver actor selection with greedy approach."""

import random
from copy import deepcopy
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def get_mds_greedy(net: MultilayerNetwork) -> set[MLNetworkActor]:
    """
    Get driver actors for a network a.k.a. a minimal dominating set.

    The dominating set gets minimised before it gets returned, but it is not a
    minimal one. For more precise, but more time-consuming approach see
    `get_mds_locimpr`.

    :param net: network to obtain minimal dominating set for
    :return: (sub)minimal dominating set
    """
    min_dominating_set: set[Any] = set()
    for layer in net.layers:
        min_dominating_set = _minimum_dominating_set_with_initial(
            net, layer, min_dominating_set
        )

    return {net.get_actor(actor_id) for actor_id in min_dominating_set}


def _minimum_dominating_set_with_initial(
    net: MultilayerNetwork, layer: str, initial_set: set[Any]
) -> set[Any]:
    """
    Return a dominating set that includes the initial set.

    :param net: multilayer network
    :param layer: name of the layer to find dominating set for
    :param initial_set: initially found dominating set
    :return: `initial_set` enhanced by dominating nodes from the current layer
    """
    actor_ids = [x.actor_id for x in net.get_actors()]
    random.shuffle(actor_ids)
    if not set(initial_set).issubset(set(actor_ids)):
        raise ValueError("Initial set must be a subset of net's actors")

    dominating_set = set(initial_set)

    net_layer = net.layers[layer]
    isolated = set(actor_ids) - set(net_layer.nodes())
    dominating_set = dominating_set | isolated
    dominated = deepcopy(dominating_set)

    layer_nodes = list(net_layer.nodes())
    random.shuffle(layer_nodes)

    for node_u in dominating_set:
        if node_u in net_layer.nodes:
            dominated.update(net_layer[node_u])

    while len(dominated) < len(net):
        node_dominate_nb = {
            x: len(set(net_layer[x]) - dominated) for x in layer_nodes
        }

        # if current dominated set cannot be enhanced anymore and there're
        # still nondominated nodes
        if sum(node_dominate_nb.values()) == 0:
            to_dominate = set(net[layer].nodes).difference(dominated)
            return dominating_set.union(to_dominate)

        # Choose a node which dominates the maximum number of undominated nodes
        node_u = max(
            node_dominate_nb.keys(), key=lambda x: node_dominate_nb[x]
        )
        dominating_set.add(node_u)
        dominated.add(node_u)
        dominated.update(net_layer[node_u])

    return dominating_set
