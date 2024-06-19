# Copyright (c) 2024 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Clique-based influence maximisation algorithm."""

# pylint: disable=C0103

from typing import Any

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


def _cim_maximal_cliques(
    net: nx.Graph, filter_cliques: bool = True
) -> list[set[Any]]:
    """
    Get maximal cliques for given single-layered graph.

    :param net: network to find cliques for
    :param filter_cliques: if true then weird filtering is applied according to
        example provided by authors of the method
    :return: list of maximal cliques as sets of nodes' ids
    """
    cliques_raw = [set(c) for c in nx.find_cliques(net)]

    if filter_cliques:
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


def _sort_clique_by_degree(
    clique: set[Any], degrees: dict[Any, int]
) -> list[Any]:
    """Sort nodes in clique by their degrees."""
    return sorted(list(clique), key=lambda x: degrees[x], reverse=True)


def _update_seed_set(
    seed_set: list[Any], clique: list[Any], update_by: int
) -> tuple[list[Any], int]:
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
    G: nx.Graph, K: int, filter_cliques: bool = False
) -> list[Any]:
    """
    Clique-based influence maximisation algorithm.

    The routine was published by Venkatakrishna Rao, Mahender Katukuri, and
    Maheswari Jagarapu in CIM: clique-based heuristic for finding influential
    nodes in multilayer networks (https://doi.org/10.1007/s10489-021-02656-0)
    which was published in "Applied Intelligence", 2022, 52:5173-5184.
    The method was optimised by the authors of Network Diffusion and deprived
    of ambiguities.

    :param G: a network to compute seeds for
    :param K: target size of the seed set
    :param filter_cliques: a flag wether to filter out cliques according to
        authors of the method; if true all cliques of size 1 and some cliques
        of size 2 will be discarded (even if they autonomous connected
        components).

    :return: a list of nodes that are chosen as seeds
    """
    if K > len(G.nodes):
        raise ValueError("Nb of seeds cannot be > than nb of nodes in graph!")

    # input data to the algorithm
    max_cliques = _cim_maximal_cliques(net=G, filter_cliques=filter_cliques)
    degrees = nx.degree(G)

    # output contaier
    seeds: list[Any] = []

    # sort all the cliques in descending order based on size
    _mcs = [_sort_clique_by_degree(mc, degrees) for mc in max_cliques]
    sorted_mcs = sorted(
        _mcs, key=lambda x: len(x), reverse=True  # pylint: disable=W0108
    )

    # until budget is spent pick the very fist nodes from all cliques that are
    # not already in the seed set
    while len(seeds) < K:
        for mc in sorted_mcs:
            seeds, _ = _update_seed_set(seed_set=seeds, clique=mc, update_by=1)
            if len(seeds) == K:
                break

    return seeds


class CIMSeedSelector(BaseSeedSelector):
    """Seed selector based on Clique-based influence maximisation algorithm."""

    def _calculate_ranking_list(self, graph: nx.Graph) -> list[Any]:
        """
        Create a ranking of nodes with Clique-based influence maximisation alg.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ranking = clique_influence_maximization(
            G=graph, K=len(graph.nodes), filter_cliques=False
        )
        if len(ranking) != len(graph.nodes):  # that's a sanity check
            raise ValueError
        return ranking

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tClique-based Influence Maximisation\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> list[MLNetworkActor]:
        """Compute ranking for actors."""
        return node_to_actor_ranking(super().nodewise(net), net)
