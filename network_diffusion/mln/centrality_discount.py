# Copyright (c) 2023 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Implementation of actors arnkings basing on Degree Discount algorithm."""

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import (
    all_neighbors,
    degree,
    neighbourhood_size,
)
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def _argmax_centrality_dict(centr_dict: dict[Any, int]) -> list[Any]:
    """Get key(s) for maximal value in the dict."""
    if len(centr_dict) == 0:
        return []
    max_value = max(centr_dict.values())
    max_keys = [key for key, val in centr_dict.items() if val == max_value]
    return max_keys


def degree_discount_networkx(net: nx.Graph, k: int) -> list[Any]:
    """
    Degree Discount Heuristic for NetworkX Graph.

    Routine was invented by W. Chen, Y. Wang, S. Yang in "Efficient Influence
    Maximization in Social Networks", (doi.org/10.1145/1557019.1557047) wchich
    was published in KDD 09: Proceedings of the 15th ACM SIGKDD international
    conference on Knowledge discovery and data mining, 2009, p. 199-208. This
    implementation was slightly modified to be independent on the Independent
    Cascade Model property - a propagation probability, but the idea stays the
    same.

    :param net: a network to compute seed ranking for
    :param k: target size of the seed set

    :return: a list of nodes ordered descending (the higher node in the list,
        more central is)
    """
    seed_list: list[Any] = []
    degrees = dict(nx.degree(net))
    while len(seed_list) < k:
        max_degree_node = _argmax_centrality_dict(degrees)[0]
        seed_list.append(max_degree_node)
        del degrees[max_degree_node]
        for neighbour in net.neighbors(max_degree_node):
            if degrees.get(neighbour):
                degrees[neighbour] -= 1
    assert len(seed_list) == k
    return seed_list


def _centrality_discount_mln(
    net: MultilayerNetwork, k: int, centr_dict: dict[MLNetworkActor, int]
) -> list[MLNetworkActor]:
    """
    Rank actors using discounting algo.

    Ranking is processed according to discounting centrality values basing on
    neighbours of picked actors, i.e. if the actor is already in the seed list,
    its neighbours (that are not in the seed list) have their centrality value
    decreased by one. This method is sensible only for degree and neighbourhood
    size.
    """
    if k > len(net):
        raise ValueError("Nb of seeds to be selscted > than number of actors!")
    seed_list: list[MLNetworkActor] = []
    while len(seed_list) < k:
        max_centr_actor = _argmax_centrality_dict(centr_dict)[0]
        seed_list.append(max_centr_actor)
        del centr_dict[max_centr_actor]
        for neighbour in all_neighbors(net=net, actor=max_centr_actor):
            if centr_dict.get(neighbour):
                centr_dict[neighbour] -= 1
    assert len(seed_list) == k
    return seed_list


def degree_discount(net: MultilayerNetwork, k: int) -> list[MLNetworkActor]:
    """
    Degree Discount Heuristic for MultilayerNetwork.

    Routine was invented by W. Chen, Y. Wang, S. Yang in "Efficient Influence
    Maximization in Social Networks", (doi.org/10.1145/1557019.1557047). This
    implementation was slightly modified to be independent on the Independent
    Cascade Model property - a propagation probability, and we have replaced
    degree by the algo sensible for multilayer networks. Anyway, the idea stays
    the same.

    :param net: a network to compute seed ranking for
    :param k: target size of the seed set

    :return: a list of actors ordered descending (the higher actor in the list,
        more central is)
    """
    degrees = degree(net=net)
    return _centrality_discount_mln(net=net, k=k, centr_dict=degrees)


def neighbourhood_size_discount(
    net: MultilayerNetwork, k: int
) -> list[MLNetworkActor]:
    """
    Neighbourhood Size Discount Heuristic for MultilayerNetwork.

    Routine was invented by W. Chen, Y. Wang, S. Yang in "Efficient Influence
    Maximization in Social Networks", (doi.org/10.1145/1557019.1557047). This
    implementation was slightly modified to be independent on the Independent
    Cascade Model property - a propagation probability, and we have replaced
    degree by neighbourhood size. Anyway, the idea stays the same.

    :param net: a network to compute seed ranking for
    :param k: target size of the seed set

    :return: a list of actors ordered descending (the higher actor in the list,
        more central is)
    """
    nsizes = neighbourhood_size(net=net)
    return _centrality_discount_mln(net=net, k=k, centr_dict=nsizes)
