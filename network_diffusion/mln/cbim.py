"""Community based influence maximization algorithm."""

import random
from typing import Any

import networkx as nx
import numpy as np


def compute_seed_quotas(
    G: nx.Graph, communities: list[Any], num_seeds: int
) -> list[int]:
    """
    Compute number of seeds to be chosen from communities according to budget.

    We would like to select <num_seeds> from the network <G>. This function
    will compute how many seeds take from each community basing on their sizes
    compared to size of the <G>. There is a condition: communities should be
    separable, i.e. each node must be in only one community.
    """
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


def get_katz_and_select_seeds(
    final_communities: list[list[Any]], G: nx.Graph, seed_budget: int
) -> list[Any]:
    """Finding Katz centrality coeficient and seed node selection."""

    # 1. Initialize the Seed node set (SN) and Community set (CS)
    SN = np.zeros(len(final_communities))
    CS = final_communities.copy()

    # 2. Calculate Katz centrality for each node in each community
    katzCentralityList = []
    for C in CS:
        katzCentralityDict = nx.katz_centrality(
            G.subgraph(C), alpha=0.1, beta=1.0, max_iter=100, weight="weight"
        )
        katzCentralityList.append(
            [(k, v) for k, v in katzCentralityDict.items()]
        )

    # 3. Sort all the nodes in each community based on Katz Centrality Coeficient in descending order
    for community in katzCentralityList:
        community.sort(key=lambda a: a[1], reverse=True)
    print(
        "Katz Centrality coeficients for every node: "
        + str(katzCentralityList)
    )

    # 4. Calculate required seed nodes from each community in quota based approach
    quotaList = compute_seed_quotas(
        G=G, communities=final_communities, num_seeds=seed_budget
    )
    print("Quotas for every community: " + str(quotaList))

    # 5. Select the quota number of highest Katz centrality coeficient nodes as seed nodes from each community (Ci)
    SN = np.zeros(int(sum(quotaList)))
    i = 0
    j = 0
    for C in CS:
        for community in katzCentralityList:
            for node in community:
                if C.count(node[0]) > 0 and quotaList[i] > 0:
                    quotaList[i] -= 1
                    SN[j] = node[0]
                    j += 1
        i += 1

    print("Community Set (CS): " + str(CS))
    print("Seed Nodes: " + str(SN))

    return SN


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
        final_communities=final_communities, G=net, seed_budget=seedNodeAmount
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


if __name__ == "__main__":
    net = get_toy_network()
    mergingIndexThreshold = 0.1
    seedNodeAmount = 4
    seeds = cbim(
        net=net,
        mergingIndexThreshold=mergingIndexThreshold,
        seedNodeAmount=seedNodeAmount,
    )
    print(
        f"nodes nb: {net.number_of_nodes()}, edges nb: {net.number_of_edges()}"
    )
    print(f"parameters: k: {seedNodeAmount}, δ: {mergingIndexThreshold}")
    print(f"seeds: {seeds}")
