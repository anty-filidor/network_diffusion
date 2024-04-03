"""Clique-based influence maximisation algorithm."""

from typing import Any, Dict, List, Set, Tuple

import networkx as nx


def cim_maximal_cliques(net: nx.Graph, filter: bool = True) -> List[Set[Any]]:
    """
    Get maximal cliques for given single-layered graph.

    :param net: network to find cliques for
    :param filter: if true then weird filtering is applied according to example
        provided by authors of the method
    :return: list of maximal cliques as sets of nodes' ids
    """
    cliques_raw = [set(c) for c in nx.find_cliques(net)]

    if filter:
        components = nx.connected_components(net)
        cliques_filtered = []
        for clique in cliques_raw:
            if len(clique) == 1:
                continue
            if len(clique) == 2 and clique not in components:
                continue
            cliques_filtered.append(clique)
        return cliques_filtered

    return cliques_raw


def sort_clique_by_degree(
    clique: Set[Any], degrees: Dict[Any, int]
) -> List[Any]:
    """Sort nodes in clique by their degrees."""
    return sorted(list(clique), key=lambda x: degrees[x], reverse=True)


def update_seed_set(
    seed_set: List[Any], clique: List[Any], update_by: int
) -> Tuple[List[Any], int]:
    """
    Add <update_by> first nodes (without duplicates) from clique to seed_set.

    As result return updated seed set and a number of nodes, by which seed set
    was eventually updated (i.e. in case of duplicates in clique and seed_set).
    """
    init_len_seed_set = len(seed_set)
    clique[:] = [node for node in clique if node not in seed_set]
    seed_set.extend(clique[:update_by])
    updated_by = len(seed_set) - init_len_seed_set
    return seed_set, updated_by


def clique_influence_maximization(
    G: nx.Graph, K: int, filter: bool = False
) -> List[Any]:
    """
    Clique-based influence maximisation algorithm.

    The routine was published by Venkatakrishna Rao, Mahender Katukuri, and
    Maheswari Jagarapu in CIM: clique-based heuristic for finding influential
    nodes in multilayer networks (https://doi.org/10.1007/s10489-021-02656-0)
    which was published in "Applied Intelligence", 2022, 52:5173-5184.
    The method was optimised by the author of Network Diffusion and deprived of
    ambiguities.

    :param G: a network to compute seeds for
    :param K: target number of the seed set
    :param filter: a flag wether to filter out cliques according to authors of
        the method; if true all cliques of size 1 and some cliques of size 2
        will be discarded (even if they autonomous connected components)

    :return: a list of nodes that are chosen as seeds
    """
    if K > len(G.nodes):
        raise ValueError("Nb of seeds cannot be > than nb of nodes in graph!")

    # input data to the algorithm
    max_cliques = cim_maximal_cliques(net=G, filter=filter)
    degrees = nx.degree(G)

    # output contaier
    seeds: List[Any] = []

    # sort all the cliques in descending order based on size
    _mcs = [sort_clique_by_degree(mc, degrees) for mc in max_cliques]
    sorted_mcs = sorted(_mcs, key=lambda x: len(x), reverse=True)

    # until budget is spent pick the very fist nodes from all cliques that are
    # not already in the seed set
    while len(seeds) < K:
        for mc in sorted_mcs:
            seeds, _ = update_seed_set(seed_set=seeds, clique=mc, update_by=1)
            if len(seeds) == K:
                break

    return seeds
