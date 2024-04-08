# https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/weic-kdd09_influence.pdf


# fore each degree value (start by he highest)
# if for this dv there is only one node assigned add it to the ranking list
# otherwise take all nodes with this degree
# for each node:
# for each neighbour of node:
# if neighbour is a neighbour of any node already added to the ranking list
# decrease degree of the node by one
# sort nodes according to computed degrees and add them in this order to the ranking list


from typing import Any, Dict, List

import networkx as nx

K = 10  # budget
G = nx.les_miserables_graph()
p = 0.6


def argmax_degrees_dict(degrees_dict: Dict[Any, int]) -> List[Any]:
    if len(degrees_dict) == 0:
        return []
    max_value = max(degrees_dict.values())
    max_keys = [key for key, val in degrees_dict.items() if val == max_value]
    return max_keys


def degree_discount(G: nx.Graph, K: int, p: float) -> List[Any]:
    S = []
    dd_v = dict(nx.degree(G))  # degrees of all nodes
    d_v = dd_v.copy()  # degrees of all nodes
    t_v = {
        node_id: 0 for node_id in G.nodes
    }  # number of neighboursof each node selected to seed list

    for _ in range(K):
        # get the node with the highest degree
        u = ...
        S.append(u)
        for v in nx.neighbors(G, u):
            if v in S:
                continue
            t_v[v] += 1
            # ddv = dv − 2tv − (dv − tv )tv p
            dd_v[v] = d_v[v] - (2 * t_v[v]) - (d_v[v] - t_v[v]) * t_v[v] * p
    return S


# def degree_discount2(G: nx.Graph, nb_seeds: int) -> List[Any]:
#     seed_list = []
#     degrees = dict(nx.degree(G))
#     while len(seed_list) < nb_seeds:
#         max_degree_nodes = argmax_degrees_dict(degrees)
#         for node in max_degree_nodes:
#             if node in seed_list:
#                 continue
#             seed_list.append(node)
#             del degrees[node]
#             for neighbour in G.neighbors(node):
#                 if degrees.get(neighbour):
#                     degrees[neighbour] -= 1
#             break
#     assert len(seed_list) == nb_seeds
#     return seed_list


def degree_discount2(G: nx.Graph, nb_seeds: int) -> List[Any]:
    seed_list = []
    degrees = dict(nx.degree(G))
    while len(seed_list) < nb_seeds:
        max_degree_node = argmax_degrees_dict(degrees)[0]
        seed_list.append(max_degree_node)
        del degrees[max_degree_node]
        for neighbour in G.neighbors(max_degree_node):
            if degrees.get(neighbour):
                degrees[neighbour] -= 1
    assert len(seed_list) == nb_seeds
    return seed_list
