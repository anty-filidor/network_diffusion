from dataclasses import dataclass
from typing import Any, Dict, List

import networkx as nx
import pandas as pd

import network_diffusion as nd


def get_toy_network_kpp_shell() -> nx.Graph:
    """
    Get a toy network that was used by the authors of CIM method.

    The paper is here: https://doi.org/10.1007/s10489-021-02656-0
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

    return nd.MultilayerNetwork.from_nx_layers([layer_1], ["l1"])


@dataclass
class KPPSNode:
    """K++ Shell aux class for state of node."""

    node_id: any
    reward_points: int
    shell: int

    def to_list(self):
        return [self.shell, self.node_id, self.reward_points]

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "shell": self.shell,
            "reward_points": self.reward_points,
        }


def _kppshell_single_community(community_subgraph: nx.Graph) -> List[Dict[str, KPPSNode]]:

    degrees = dict(nx.degree(community_subgraph))
    nodes = {node_id: KPPSNode(node_id, 0, degree) for node_id, degree in degrees.items()}
    bucket_list: List[KPPSNode] = []

    # iterate until there is still a node to be picked
    while len(nodes) >= 1:
        minimal_degree = min(nodes.values(), key=lambda x: x.shell).shell
        picked_curr_nodes = []
        for node_id in community_subgraph.nodes:

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
            for neighbour_id in community_subgraph.neighbors(node_id):
                if nodes.get(neighbour_id):
                    nodes[neighbour_id].shell -= 1
                    nodes[neighbour_id].reward_points += 1

    # return bucket list in form of: bucket id, node id and reward points
    return [b.to_dict() for b in bucket_list]


def kppshell_decomposition(G: nx.Graph):
    communities = list(nx.community.label_propagation_communities(G))
    bucket_lists = []
    for community in communities:
        bucket_list = _kppshell_single_community(
            community_subgraph=G.subgraph(community)
        )
        bucket_lists.append(bucket_list)
    return bucket_lists


def compute_seed_quotas(
    G: nx.Graph, communities: List[Any], num_seeds: int
) -> List[int]:
    """
    Compute number of seeds to be chosen from communities according to budget.

    There is a condition: communities should be separable, i.e. each node must
    to be in only one community.
    """
    if num_seeds > len(G.nodes):
        raise ValueError("Number of seeds cannot be > number of nodes!")
    quotas = []

    # compute fractions of communities to use as seeds
    for communiy in communities:
        quo = num_seeds * len(communiy) / len(G.nodes)
        quotas.append(int(quo))
    # print("raw quotas:", quotas, "com-s:", [len(c) for c in communities])

    # quantisation of computed quotas according to strategy: first increase
    # quotas in communities that were skipped and if there is no such community
    # increase quotas in the larges communities; here we assume that community
    # is at least 1 node large
    while sum(quotas) < num_seeds:
        if quotas[quotas.index(min(quotas))] == 0:
            quotas[quotas.index(min(quotas))] += 1
        else:
            diffs = [len(com) - quo for quo, com in zip(quotas, communities)]
            _quotas = quotas.copy()
            while True:
                quota_to_increase = max(_quotas)
                if diffs[quotas.index(quota_to_increase)] > 0:
                    break
                del _quotas[quota_to_increase]
            quotas[quotas.index(quota_to_increase)] += 1
    # print("balanced quotas:", quotas, "com-s:", [len(c) for c in communities])

    # sanity check - the function is still in development mode
    if sum(quotas) != num_seeds:
        raise ArithmeticError("Error in the function!")
    for quo, com in zip(quotas, communities):
        if quo > len(com):
            raise ArithmeticError("Error in the function!")

    return quotas


def kppshell_seed_selection(G: nx.Graph, num_seeds: int):
    shells = kppshell_decomposition(G)
    quotas_in_shells = compute_seed_quotas(
        G=G,
        communities=[[node["node_id"] for node in shell] for shell in shells],
        num_seeds=num_seeds,
    )
    seeds_ranked = []

    for quota, decomposed_community in zip(quotas_in_shells, shells):
        df = (
            pd.DataFrame(decomposed_community)
            .sort_values(["shell", "reward_points"], ascending=[False, False])
            .reset_index(drop=True)
        )
        # print(quota)
        # print(df)
        # print("\n\n\n")
        seeds_ranked.extend(df.iloc[:quota]["node_id"].to_list())

    return seeds_ranked


# 𝐶1 = {(𝐵1, 11, 0), (𝐵2, 5, 1)(𝐵2, 2, 0)(𝐵3, 1, 2), (𝐵3, 7, 2), (𝐵3, 3, 0), (𝐵3, 4, 0)}
# 𝐶2 = {(𝐵1, 8, 0), (𝐵2, 6, 2), (𝐵2, 9, 0), (𝐵2, 10, 0), (𝐵2, 12, 0)}.

# ground truth data for toy network
# [[{'node_id': 11, 'shell': 1, 'reward_points': 0},
#   {'node_id': 2, 'shell': 2, 'reward_points': 0},
#   {'node_id': 5, 'shell': 2, 'reward_points': 1},
#   {'node_id': 1, 'shell': 3, 'reward_points': 2},
#   {'node_id': 3, 'shell': 3, 'reward_points': 0},
#   {'node_id': 4, 'shell': 3, 'reward_points': 0},
#   {'node_id': 7, 'shell': 3, 'reward_points': 2}],
#  [{'node_id': 8, 'shell': 1, 'reward_points': 0},
#   {'node_id': 6, 'shell': 2, 'reward_points': 1},
#   {'node_id': 9, 'shell': 2, 'reward_points': 0},
#   {'node_id': 10, 'shell': 2, 'reward_points': 0},
#   {'node_id': 12, 'shell': 2, 'reward_points': 0}]]
