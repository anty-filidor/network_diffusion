"""Implementation of seed selection basing on Degree Discount algorithm."""

from typing import Any, Dict, List

import networkx as nx


def _argmax_degrees_dict(degrees_dict: Dict[Any, int]) -> List[Any]:
    """Helper function to get key(s) for maximal value in the dict."""
    if len(degrees_dict) == 0:
        return []
    max_value = max(degrees_dict.values())
    max_keys = [key for key, val in degrees_dict.items() if val == max_value]
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
        max_degree_node = _argmax_degrees_dict(degrees)[0]
        seed_list.append(max_degree_node)
        del degrees[max_degree_node]
        for neighbour in G.neighbors(max_degree_node):
            if degrees.get(neighbour):
                degrees[neighbour] -= 1
    assert len(seed_list) == K
    return seed_list
