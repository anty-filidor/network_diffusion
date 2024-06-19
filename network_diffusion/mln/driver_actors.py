# Copyright (c) 2023 by Yu-Xuan Qi, Mingshan Jia, Katarzyna Musial, Michał
# Czuba.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Script with functions for driver actor selections."""

from copy import deepcopy
from typing import Any

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def compute_driver_actors(net: MultilayerNetwork) -> list[MLNetworkActor]:
    """Return driver actors for a given network."""
    min_dominating_set: set[Any] = set()
    for layer in net.layers:
        min_dominating_set = minimum_dominating_set_with_initial(
            net, layer, min_dominating_set
        )

    return [net.get_actor(actor_id) for actor_id in min_dominating_set]


def minimum_dominating_set_with_initial(
    net: MultilayerNetwork, layer: str, initial_set: set[Any]
) -> set[int]:
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
    dominated = deepcopy(dominating_set)

    for node_u in dominating_set:
        if node_u in net_layer.nodes:
            dominated.update(net_layer[node_u])

    while len(dominated) < len(net):
        # Choose a node which dominates the maximum number of undominated nodes
        node_u = max(
            net_layer.nodes(), key=lambda x: len(set(net_layer[x]) - dominated)
        )
        dominating_set.add(node_u)
        dominated.add(node_u)
        dominated.update(net_layer[node_u])

    return dominating_set
