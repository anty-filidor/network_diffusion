"""Implementation of seed selection basing on Degree Discount algorithm."""

from typing import Any, Dict, List

from network_diffusion.mln.functions import neighbourhood_size, degree, all_neighbors
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
import networkx as nx


def _argmax_centrality_dict(centr_dict: Dict[Any, int]) -> List[Any]:
    """Helper function to get key(s) for maximal value in the dict."""
    if len(centr_dict) == 0:
        return []
    max_value = max(centr_dict.values())
    max_keys = [key for key, val in centr_dict.items() if val == max_value]
    return max_keys


def degree_discount(G: nx.Graph, K: int) -> List[Any]:
    """
    Degree Discount Heuristic.

    Routine was invented by W. Chen, Y. Wang, S. Yang in "Efficient Influence
    Maximization in Social Networks", (doi.org/10.1145/1557019.1557047) wchich
    was published in KDD 09: Proceedings of the 15th ACM SIGKDD international
    conference on Knowledge discovery and data mining, 2009, pages 199-208. This
    implementation was slightly modified to be independent on the Independent
    Cascade Model property - a propagation probability, but the idea stays the
    same.

    :param G: a network to compute seed ranking for
    :param K: target size of the seed set

    :return: a list of nodes that are chosen as seeds
    """
    seed_list = []
    degrees = dict(nx.degree(G))
    while len(seed_list) < K:
        max_degree_node = _argmax_centrality_dict(degrees)[0]
        seed_list.append(max_degree_node)
        del degrees[max_degree_node]
        for neighbour in G.neighbors(max_degree_node):
            if degrees.get(neighbour):
                degrees[neighbour] -= 1
    assert len(seed_list) == K
    return seed_list


def _centrality_discount_mln(
    G: MultilayerNetwork, K: int, centr_dict: Dict[MLNetworkActor, int]
) -> List[MLNetworkActor]:
    seed_list = []
    print({a.actor_id: v for a, v in centr_dict.items()})
    while len(seed_list) < K:
        max_centr_actor = _argmax_centrality_dict(centr_dict)[0]
        seed_list.append(max_centr_actor)
        del centr_dict[max_centr_actor]
        for neighbour in all_neighbors(net=G, actor=max_centr_actor):
            if centr_dict.get(neighbour):
                centr_dict[neighbour] -= 1
    assert len(seed_list) == K
    return seed_list


def degree_discount_mln(G: MultilayerNetwork, K: int) -> List[MLNetworkActor]:
    degrees = degree(net=G)
    return _centrality_discount_mln(G=G, K=K, centr_dict=degrees)


def neighbourhood_size_discount(G: MultilayerNetwork, K: int) -> List[MLNetworkActor]:
    nsizes = neighbourhood_size(net=G)
    return _centrality_discount_mln(G=G, K=K, centr_dict=nsizes)
