from dataclasses import dataclass

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
class RewardedNode:
    node_id: any
    reward_points: int
    degree: int

    @classmethod
    def make_raw(cls, node_id: int) -> "RewardedNode":
        return cls(node_id=node_id, reward_points=0, degree=0)

    def __hash__(self):
        return self.node_id

    def to_list(self):
        return [self.degree, self.node_id, self.reward_points]


def k_pp_shell(G: nx.Graph):

    communities = list(nx.community.label_propagation_communities(G))

    bbls = []

    for community in communities:

        GG = G.subgraph(community)
        degrees = dict(nx.degree(GG))
        nodes = {
            node_id: RewardedNode(
                node_id=node_id, reward_points=0, degree=degree
            )
            for node_id, degree in degrees.items()
        }
        BL = []

        while len(nodes) >= 1:
            minimal_degree = min(nodes.values(), key=lambda x: x.degree).degree
            print(minimal_degree, BL)
            for node_id in community:

                if node_id in [b.node_id for b in BL]:
                    continue

                node_obj = nodes[node_id]
                if node_obj.degree == minimal_degree:
                    BL.append(node_obj)
                    del nodes[node_id]

            # update degrees and reward points for neighbours of chosen nodes
            for node in BL:
                for neighbour in GG.neighbors(node.node_id):
                    if nodes.get(neighbour):
                        nodes[neighbour].degree -= 1
                        nodes[neighbour].reward_points += 1

        bbls.append([b.to_list() for b in BL])

    return bbls


# 𝐶1 = {(𝐵1, 11, 0), (𝐵2, 5, 1)(𝐵2, 2, 0)(𝐵3, 1, 2), (𝐵3, 7, 2), (𝐵3, 3, 0), (𝐵3, 4, 0)}
# 𝐶2 = {(𝐵1, 8, 0), (𝐵2, 6, 2), (𝐵2, 9, 0), (𝐵2, 10, 0), (𝐵2, 12, 0)}.
