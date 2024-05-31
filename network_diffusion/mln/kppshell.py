# Copyright 2024 by Michał Czuba, Piotr Bródka. All Rights Reserved.
#
# This file is part of Network Diffusion.
#
# Network Diffusion is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Network Diffusion is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# Network Diffusion. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""K++ Shell decomposition functions."""

# pylint: disable=C0103

from dataclasses import dataclass
from typing import Any, Dict, List, Set

import networkx as nx
import pandas as pd

from network_diffusion.mln.mlnetwork import MultilayerNetwork


def get_toy_network_kppshell() -> MultilayerNetwork:
    """
    Get a toy network that was used by the authors of K++ Shell method.

    The paper is here: https://doi.org/10.1016/j.comnet.2023.109916
    """
    actors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    layer_1 = nx.Graph(
        (
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 7],
            [2, 7],
            [3, 4],
            [3, 7],
            [4, 7],
            [5, 7],
            [5, 6],
            [5, 11],
            [6, 8],
            [6, 9],
            [6, 10],
            [9, 12],
            [10, 12],
        )
    )
    layer_1.add_nodes_from(actors)

    return MultilayerNetwork.from_nx_layers([layer_1], ["l1"])


@dataclass
class KPPSNode:
    """K++ Shell auxiliary class for keep state of node."""

    node_id: Any
    shell: int
    reward_points: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialise object into dictionary."""
        return {
            "node_id": self.node_id,
            "shell": self.shell,
            "reward_points": self.reward_points,
        }


def _kppshell_community(
    community_graph: nx.Graph,
) -> List[Dict[str, KPPSNode]]:
    """
    K++ Shell decomposition of a single community of the graph.

    As a result return a list of nodes with shells that they were assigned to
    and reward points they gained during decomposition.
    """
    degrees = dict(nx.degree(community_graph))
    nodes = {
        node_id: KPPSNode(node_id, degree, 0)
        for node_id, degree in degrees.items()
    }
    bucket_list: List[KPPSNode] = []

    # iterate until there is still a node to be picked
    while len(nodes) >= 1:
        minimal_degree = min(nodes.values(), key=lambda x: x.shell).shell
        picked_curr_nodes = []
        for node_id in community_graph.nodes:

            # if node has been already removed
            if node_id in [b.node_id for b in bucket_list]:
                continue

            # it current degree of the node is minimal add it to BL and remove
            if nodes[node_id].shell == minimal_degree:
                bucket_list.append(nodes[node_id])
                picked_curr_nodes.append(node_id)
                del nodes[node_id]

        # update state of still not picked neighbours of chosen nodes
        for node_id in picked_curr_nodes:
            for neighbour_id in community_graph.neighbors(node_id):
                if nodes.get(neighbour_id):
                    nodes[neighbour_id].shell -= 1
                    nodes[neighbour_id].reward_points += 1

    # return bucket list in form of: bucket id, node id and reward points
    assert len(bucket_list) == len(community_graph.nodes)
    return [b.to_dict() for b in bucket_list]


def kppshell_decomposition(G: nx.Graph) -> List[List[Dict[str, Any]]]:
    """
    Decompose network according to K++ shell routine.

    The routine was published by Venkatakrishna Rao, C. Ravindranath Chowdary
    in "K++ Shell: Influence maximization in multilayer networks using
    community detection" (https://doi.org/10.1016/j.comnet.2023.109916)
    which was published in "Computer Networks", 2023, Volume 234. The method
    was optimised by the authors of Network Diffusion and deprived of
    ambiguities. We also made an assumption that during communities detection
    every node can be assigned only to one community. Another assumption was
    that we accept communities that consist of at least one node. These
    conditions were not considered in the original paper.

    :param G: a network to be decomposed

    :return: a list of communities detected in the network in which all nodes
        are represented as dict with thei IDs, shells they are assigned to and
        reward points their gained during decomposition.
    """
    communities = list(nx.community.label_propagation_communities(G))
    assert sum(len(c) for c in communities) == len(G.nodes)
    bucket_lists = []
    for community in communities:
        bucket_list = _kppshell_community(
            community_graph=G.subgraph(community)
        )
        bucket_lists.append(bucket_list)
    return bucket_lists


def compute_seed_quotas(
    G: nx.Graph, communities: List[Any], num_seeds: int
) -> List[int]:
    """
    Compute number of seeds to be chosen from communities according to budget.

    We would like to select <num_seeds> from the network <G>. This function
    will compute how many seeds take from each community basing on their sizes
    compared to size of the <G>. There is a condition: communities should be
    separable, i.e. each node must be in only one community.
    """
    # pylint: disable=R0912
    if num_seeds > len(G.nodes):
        raise ValueError("Number of seeds cannot be > number of nodes!")
    quotas = []

    # sort communities according to their sizes and remember their indices
    # in the input matrix to return quotas in correct order
    comms_sorting_order = sorted(
        range(len(communities)),
        key=lambda k: [len(c) for c in communities][k],
        reverse=True,
    )
    comms_sorted = [communities[comm_idx] for comm_idx in comms_sorting_order]

    # compute fractions of communities to be used as seeds
    for communiy in comms_sorted:
        quo = num_seeds * len(communiy) / len(G.nodes)
        quotas.append(int(quo))
    # print("raw quotas", quotas, "com-s", [len(c) for c in comms_sorted])

    # quantisation of computed quotas according to strategy: first increase
    # quotas in communities that were skipped and if there is no such community
    # increase quotas in the largest communities that have capability to have
    # quota enlarged; here we assume that community is at least 1 node large
    while sum(quotas) < num_seeds:
        if quotas[quotas.index(min(quotas))] == 0:
            quotas[quotas.index(min(quotas))] += 1
        else:
            diffs = [len(com) - quo for quo, com in zip(quotas, comms_sorted)]
            _quotas = quotas.copy()
            while True:
                quota_to_increase = max(_quotas)
                if diffs[quotas.index(quota_to_increase)] > 0:
                    break
                del _quotas[_quotas.index(quota_to_increase)]
            quotas[quotas.index(quota_to_increase)] += 1
    # print("balanced quotas", quotas, "com-s", [len(c) for c in comms_sorted])

    # reorder quotas for communities according to their initial order
    quotas_desorted = [0] * len(communities)
    for idx_sorted, idx_original in enumerate(comms_sorting_order):
        quotas_desorted[idx_original] = quotas[idx_sorted]

    # sanity check - the function is still in development mode
    if sum(quotas) != num_seeds:
        raise ArithmeticError("Error in the function!")
    for quo, com in zip(quotas, comms_sorted):
        if quo > len(com):
            raise ArithmeticError("Error in the function!")
    for quo, com in zip(quotas_desorted, communities):
        if quo > len(com):
            raise ArithmeticError("Error in the function!")

    return quotas_desorted


def _select_seeds_from_kppshells(
    shells: List[List[Dict[str, Any]]], quotas: List[int]
) -> Set[Any]:
    """Select seeds according to quota and decomposed network to the shells."""
    seeds_ranked = []
    for quota, decomposed_community in zip(quotas, shells):
        df = (
            pd.DataFrame(decomposed_community)
            .sort_values(["shell", "reward_points"], ascending=[False, False])
            .reset_index(drop=True)
        )
        # print(quota)
        # print(df)
        # print("\n\n\n")
        seeds_ranked.extend(df.iloc[:quota]["node_id"].to_list())
    return set(seeds_ranked)


def kppshell_seed_selection(G: nx.Graph, num_seeds: int) -> Set[Any]:
    """
    Select <num_seeds> from <G> according to K++ Shell decomposition.

    The routine was published by Venkatakrishna Rao, C. Ravindranath Chowdary
    in "K++ Shell: Influence maximization in multilayer networks using
    community detection" (https://doi.org/10.1016/j.comnet.2023.109916)
    which was published in "Computer Networks", 2023, Volume 234. The method
    was optimised by the authors of Network Diffusion and deprived of
    ambiguities.

    :param G: a network to pick seeds from
    :param num_seeds: number of seeds to pick, in form of number of nodes

    :return: a set of nodes picked
    """
    shells = kppshell_decomposition(G)
    quotas_in_shells = compute_seed_quotas(
        G=G,
        communities=[[node["node_id"] for node in shell] for shell in shells],
        num_seeds=num_seeds,
    )
    return _select_seeds_from_kppshells(shells=shells, quotas=quotas_in_shells)


def kppshell_seed_ranking(G: nx.Graph) -> List[Any]:
    """
    Rank all nodes from <G> according to K++ Shell decomposition.

    The routine is a modification of function kppshell_seed_selection so that
    not a given fraction of most influential nodes is returned but all of them.

    :param G: a network to create ranking of all nodes from

    :return: an ordered ranking of nodes (most influential are at the beggining
        of the list)
    """
    shells = kppshell_decomposition(G)
    communities = [[node["node_id"] for node in shell] for shell in shells]
    ranking = []
    for i in range(1, len(G.nodes) + 1):
        quotas_in_shells = compute_seed_quotas(G, communities, i)
        seeds = _select_seeds_from_kppshells(shells, quotas_in_shells)
        for seed in seeds:
            if seed not in ranking:
                ranking.append(seed)
    return ranking
