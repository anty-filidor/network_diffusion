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

"""Community based influence maximization algorithm."""

import random
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from network_diffusion.mln.kppshell import compute_seed_quotas


def dsc(G: nx.graph, u: Any, v: Any) -> float:
    """Compute Dice Similarity Coefficient between two nodes in the graph."""
    return (2 * len(list(nx.common_neighbors(G, u, v)))) / (
        len(list(G.neighbors(u))) + len(list(G.neighbors(v)))
    )


def init_communities(G: nx.Graph) -> list[list[Any]]:
    """Phase 1 - Initial communities detection."""

    # 1. Initialize the variables 'NS' and 'CSinit'
    NS = list(G.nodes)
    CSinit = []

    while len(NS) > 0:

        # 2. Select the highest degree nodes 'Dv'
        degreeList = G.degree(weight="weight")
        highestDegree = 0
        highestDegreeNodes = []
        for node in degreeList:
            if node[1] == highestDegree and NS.count(node[0]) > 0:
                highestDegreeNodes.append(node)
            elif node[1] > highestDegree and NS.count(node[0]) > 0:
                highestDegree = node[1]
                highestDegreeNodes = []
                highestDegreeNodes.append(node)

        # 3. Randomly select a node 'v' from 'Dv'
        print("The highest degree nodes are: " + str(highestDegreeNodes))
        v, _ = random.choice(highestDegreeNodes)
        print("The chosen highest degree (v) node is: " + str(v))

        # 4. Get the most similar neighbours of 'v' using Dice neighbour similarity
        highestDSC = 0.0
        highestDSCNodes = []
        for u in G.neighbors(v):
            DSC = dsc(G=G, u=u, v=v)
            if DSC == highestDSC:
                highestDSCNodes.append(u)
            elif DSC > highestDSC:
                highestDSC = DSC
                highestDSCNodes = []
                highestDSCNodes.append(u)

        # 5. Randomly select a neighbour 'sn' from the most similar neighbours list
        print("The most similar neighbour nodes are: " + str(highestDSCNodes))
        sn = random.choice(highestDSCNodes)
        print("The chosen neighbour node (sn) is: " + str(sn))

        # 6. IF 'sn' is not in any community THEN
        nodeIsPresent = False
        for C in CSinit:

            # 10. ELSE
            # 11. Find the community to which 'sn' belongs to and denote it as 'C'
            print(C)
            print(C.count(sn))
            if C.count(sn) > 0:
                # 12. Insert node 'v' into 'C' and remove node 'v' from 'NS'
                C.append(v)
                NS.remove(v)
                nodeIsPresent = True

        if not nodeIsPresent:
            # 7. Create a new community and assign 'v' and 'sn' to it
            community = [v, sn]

            # 8. Insert the new community into community structure
            CSinit.append(community)

            # 9. Remove 'v' and 'sn' from NS
            NS.remove(v)
            NS.remove(sn)

        # 13. Repeat the steps from 2 to 12, until 'NS' = null
    return CSinit


def consolide_communities(
    CSinit: list[list[Any]], G: nx.Graph, mergingIndexThreshold: float
) -> list[list[Any]]:
    """Phase 2 - Community consolidation."""
    assert 0 <= mergingIndexThreshold <= 1

    # 14. Initialize Final Communities (FC)
    FC = CSinit.copy()

    # 15. Calculate community conductance (γi) and community scale (θi) for each community 'C' in CSinit
    EoutList = np.zeros(len(CSinit))
    EinList = np.zeros(len(CSinit))
    conductanceList = np.zeros(len(CSinit))
    scaleList = np.zeros(len(CSinit))
    i = 0
    for C in CSinit:
        visitedNodes = []
        for node in C:
            visitedNodes.append(node)
            for neighbor in list(G.neighbors(node)):
                if visitedNodes.count(neighbor) == 0:
                    if C.count(neighbor) == 0:
                        EoutList[i] += G[node][neighbor]["weight"]
                    else:
                        EinList[i] += G[node][neighbor]["weight"]
        conductanceList[i] = EoutList[i] / (2 * EinList[i] + EoutList[i])
        scaleList[i] = len(C) / len(list(G.nodes))
        i += 1
    print("Community conductances: " + str(conductanceList))
    print("Community scales: " + str(scaleList))

    # 16. Calculate the merging index (ψi) for each community in CSinit.
    mergingIndexList = np.zeros(len(CSinit))
    i = 0
    for index in mergingIndexList:
        mergingIndexList[i] = conductanceList[i] * scaleList[i]
        i += 1
    print("Community merging indexes: " + str(mergingIndexList))

    # 17. Select the community with lowest merging index (Cx)
    lowestMergingIndex = (0, 0)
    mergingIndexList.sort
    i = 0
    for index in mergingIndexList:
        if mergingIndexList[i] < lowestMergingIndex[
            1
        ] or lowestMergingIndex == (0, 0):
            lowestMergingIndex = (i, mergingIndexList[i])
        i += 1

    while lowestMergingIndex[1] <= mergingIndexThreshold:
        # 17. Select the community with lowest merging index (Cx)
        lowestMergingIndex = (0, 0)
        mergingIndexList.sort
        i = 0
        for index in mergingIndexList:
            if mergingIndexList[i] < lowestMergingIndex[
                1
            ] or lowestMergingIndex == (0, 0):
                lowestMergingIndex = (i, mergingIndexList[i])
            i += 1

        print("Lowest merging index community: " + str(lowestMergingIndex[0]))

        # 18. Find the most similar community (Cy) to (Cx) and merge the two communites to form a new community (Cn)
        similarityList = np.zeros(len(CSinit))
        i = 0
        for C in FC:
            DSCsum = 0
            if i != lowestMergingIndex[0]:
                for u in CSinit[lowestMergingIndex[0]]:
                    for v in C:
                        DSCsum += (
                            2 * len(list(nx.common_neighbors(G, u, v)))
                        ) / (
                            len(list(G.neighbors(u)))
                            + len(list(G.neighbors(v)))
                        )
                similarityList[i] = DSCsum / len(CSinit[lowestMergingIndex[0]])
            i += 1
        print(
            "Communities similarity to community "
            + str(lowestMergingIndex[0])
            + ": "
            + str(similarityList)
        )

        highestSimilarity = (0, 0)
        i = 0
        for similarity in similarityList:
            if similarity > highestSimilarity[1]:
                highestSimilarity = (i, similarity)
            i += 1
        print("Most similar community: " + str(highestSimilarity[0]))
        newCommunity = FC[lowestMergingIndex[0]] + FC[highestSimilarity[0]]
        print("New community: " + str(newCommunity))

        # 19. Calculate the merging index (ψn) for new community (Cn)
        Eout = 0
        Ein = 0
        conductance = 0
        scale = len(newCommunity) / len(list(G.nodes))

        visitedNodes = []
        for node in newCommunity:
            visitedNodes.append(node)
            for neighbor in list(G.neighbors(node)):
                if visitedNodes.count(neighbor) == 0:
                    if C.count(neighbor) == 0:
                        Eout += G[node][neighbor]["weight"]
                    else:
                        Ein += G[node][neighbor]["weight"]

        conductance = Eout / (2 * Ein + Eout)

        print("New community conductance: " + str(conductance))
        print("New community scale: " + str(scale))
        mergingIndex = conductance * scale
        print("New community merging index: " + str(mergingIndex))

        # 20. Replace two communites 'Cx' and 'Cy' with new community 'Cn' in final community set (FC)
        FC[lowestMergingIndex[0]] = newCommunity
        del FC[highestSimilarity[0]]
        print("Final Communities list (FC) so far: " + str(FC))

        # 21. Repeat the process from 17 to 20 until 'ψi' > 'δ'
        lowestMergingIndex = (0, mergingIndex)

    print("Final Communities list (FC): " + str(FC))
    return FC


def _select_seeds_from_katz(kcl: list[list[dict[str, Any]]], quotas: list[int]):
    """Select top-k nodes with highest katz-centrality from each community."""
    seeds_ranked = []
    for quota, decomposed_community in zip(quotas, kcl):
        # sort all the nodes in each community based on Katz Centrality Coeficient in descending order
        df = (
            pd.DataFrame(decomposed_community)
            .sort_values(["katz_centrality"], ascending=[False])
            .reset_index(drop=True)
        )
        # select the quota number of highest Katz centrality coeficient nodes as seed nodes from each community (Ci)
        seeds_ranked.extend(df.iloc[:quota]["node_id"].to_list())
    return set(seeds_ranked)


def get_katz_and_select_seeds(G: nx.Graph, communities: list[list[Any]], num_seeds: int) -> list[Any]:
    """
    Find Katz centrality coeficient and select seed nodes.

    Implementation of Algorithm 3 from https://doi.org/10.1016/j.ins.2022.07.103
    based on code from here: https://github.com/doublejv/CBIM-Implementation.

    :param G: input graph
    :param final_communities: nodes assigned to communities
    :param seed_budget: how much nodes to choose

    :return: list of picked nodes as seeds
    """
    katz_centralities = []

    # calculate Katz centrality for each node in each community
    for C in communities:
        kc_raw = nx.katz_centrality(G.subgraph(C), alpha=0.1, beta=1.0, max_iter=100, weight="weight")
        katz_centralities.append([{"node_id": k, "katz_centrality": v} for k, v in kc_raw.items()])

    # calculate required seed nodes from each community in quota based approach
    quotas = compute_seed_quotas(G=G, communities=communities, num_seeds=num_seeds)

    # select seed set according to katz centality in each community
    return _select_seeds_from_katz(kcl=katz_centralities, quotas=quotas)


def cbim(
    net: nx.Graph, mergingIndexThreshold: float, seedNodeAmount: int
) -> list[Any]:
    initial_communities = init_communities(G=net)
    final_communities = consolide_communities(
        G=net,
        CSinit=initial_communities,
        mergingIndexThreshold=mergingIndexThreshold,
    )
    seed_set = get_katz_and_select_seeds(
        communities=final_communities, G=net, num_seeds=seedNodeAmount
    )
    return seed_set


def get_toy_network():
    adjecency_matrix = np.array(
        [
            [0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0, 3, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    )
    net = nx.from_numpy_matrix(adjecency_matrix)
    net.remove_edges_from(nx.selfloop_edges(net))
    return net
