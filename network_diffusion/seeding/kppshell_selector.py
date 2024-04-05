from dataclasses import dataclass
from typing import List

import networkx as nx

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
    degree: int  # TODO rename to shell

    def to_list(self):
        return [self.degree, self.node_id, self.reward_points]
    
    def to_dict(self):
        return {"node_id": self.node_id, "shell": self.degree, "reward_points": self.reward_points}
    

def _kppshell_single_community(community_subgraph: nx.Graph):
    
    degrees = dict(nx.degree(community_subgraph))
    nodes = {node_id: KPPSNode(node_id, 0, degree) for node_id, degree in degrees.items()}
    bucket_list: List[KPPSNode] = []

    # iterate until there is still a node to be picked
    while len(nodes) >= 1:
        minimal_degree = min(nodes.values(), key=lambda x: x.degree).degree
        picked_curr_nodes = []
        for node_id in community_subgraph.nodes:

            # if node has been already removed
            if node_id in [b.node_id for b in bucket_list]:
                continue
            
            # it current degree of the node is minimal add it to BL and remove
            if nodes[node_id].degree == minimal_degree:
                bucket_list.append(nodes[node_id])
                picked_curr_nodes.append(node_id)
                del nodes[node_id]

        # update state of still not picked neighbours of chosen nodes
        for node_id in picked_curr_nodes:
            for neighbour_id in community_subgraph.neighbors(node_id):
                if nodes.get(neighbour_id):
                    nodes[neighbour_id].degree -= 1
                    nodes[neighbour_id].reward_points += 1

    # return bucket list in form of: bucket id, node id and reward points
    return [b.to_dict() for b in bucket_list]


def kppshell_decomposition(G: nx.Graph):
    communities = list(nx.community.label_propagation_communities(G))
    bucket_lists = []
    for community in communities:
        bucket_list = _kppshell_single_community(community_subgraph=G.subgraph(community))
        bucket_lists.append(bucket_list)
    return bucket_lists


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