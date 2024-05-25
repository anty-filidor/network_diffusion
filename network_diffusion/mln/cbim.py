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

# pylint: disable=C0103, C3001

import warnings
from typing import Any, Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd
from networkx import PowerIterationFailedConvergence

from network_diffusion.mln.kppshell import compute_seed_quotas


def get_toy_network_cbim() -> nx.DiGraph:
    """
    Get a toy network.

    The network was ised by the author of intitial implementation of the CBIM
    method (https://github.com/doublejv/CBIM-Implementation).
    """
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
    net = nx.from_numpy_matrix(adjecency_matrix, create_using=nx.DiGraph)
    net.remove_edges_from(nx.selfloop_edges(net))
    return net


def _printd(statement: str, debug: bool = False) -> None:
    """Hepler function to print statements in the debug mode."""
    if debug:
        print(statement)


def _dsc(G: nx.graph, u: Any, v: Any) -> float:
    """Compute Dice Similarity Coefficient between two nodes in the graph."""
    neighbours_u = set(G.neighbors(u))
    neighbours_v = set(G.neighbors(v))
    common_neighbours = neighbours_u.copy().intersection(neighbours_v)
    if (denominator := (len(neighbours_u) + len(neighbours_v))) == 0:
        return 0
    return (2 * len(common_neighbours)) / denominator


def _gamma(nb_edges_out: int, nb_edges_in: int) -> float:
    """Calculate conductance (γi)."""
    if (denominator := 2 * nb_edges_in + nb_edges_out) == 0:
        raise ArithmeticError("Gamma cannot be computed (division by zero)!")
    return nb_edges_out / denominator


def _theta(nb_nodes_comm: int, nb_nodes_graph: int) -> float:
    """Calculate scale (θi)."""
    return nb_nodes_comm / nb_nodes_graph


def _psi(gamma: float, theta: float) -> float:
    """Calculate the merging index (ψi)."""
    return gamma * theta


def _merging_index(
    G: nx.Graph,
    community: list[Any],
    weight_attr: Optional[str] = None,
) -> float:
    """Calculate merging index for the given community of the graph."""
    visited_nodes = []
    edges_out = 0
    edges_in = 0

    if weight_attr is None:
        get_edge_weight = lambda u, v: 1  # noqa: E731
    else:
        get_edge_weight = lambda u, v: G[u][v][weight_attr]  # noqa: E731

    for node in community:
        visited_nodes.append(node)
        for neighbour in list(G.neighbors(node)):
            if visited_nodes.count(neighbour) == 0:
                if community.count(neighbour) == 0:
                    edges_out += get_edge_weight(node, neighbour)
                else:
                    edges_in += get_edge_weight(node, neighbour)

    try:
        scale = _theta(
            nb_nodes_comm=len(community), nb_nodes_graph=len(list(G.nodes))
        )
        conductance = _gamma(nb_edges_out=edges_out, nb_edges_in=edges_in)
        merging_index = _psi(gamma=conductance, theta=scale)
    except ArithmeticError:
        merging_index = 1.0

    return merging_index


def _get_extreme_val(
    val_arr: list[int | float], extreme_func: Callable
) -> tuple[int, int | float]:
    """Get the community with the lowest/highest value (whatever it is)."""
    extreme_idx = extreme_func(val_arr)
    if isinstance(extreme_idx, np.ndarray):
        extreme_idx = extreme_idx[0]
    return extreme_idx, val_arr[extreme_idx]


def _initialise_communities_in_component(
    G: nx.Graph, debug: bool, weight_attr: Optional[str] = None
) -> list[list[Any]]:
    """
    Phase 1 - Initial communities detection.

    :param G: input graph
    :param weight_attr: edge attribute to take weight from, defaults to None

    :return: initial divison of nodes into communities
    """
    # 1. Initialize the variables 'NS' and 'CSinit' and list of the degrees
    nodes_degrees = dict(G.degree(weight=weight_attr))
    initial_communities: list[list[Any]] = []

    # 11. Repeat the steps from 2 to 10, until nodes_degrees is null
    while len(nodes_degrees) > 0:

        # 2. Select the highest degree node not in initial_communities
        _printd(f"Degrees: {nodes_degrees}", debug)
        v = max(nodes_degrees, key=nodes_degrees.__getitem__)
        _printd(
            f"The chosen hi. deg. ({nodes_degrees[v]}) node is: {v}", debug
        )

        # 3a. Compute DSC similarities
        DSCs = {u: _dsc(G=G, u=u, v=v) for u in G.neighbors(v)}
        _printd(f"DSCs: {DSCs}", debug)

        # MCz: if node has no neighbours assign it to a new community
        if len(DSCs) == 0:
            _printd(f"Node {v} is assigned to the new community", debug)
            initial_communities.append([v])
            del nodes_degrees[v]
            continue

        # 3b. Get the most similar neighbours of 'v' using DSC similarity
        sn = max(DSCs, key=DSCs.__getitem__)
        _printd(f"The chosen neighbour node (sn) is: {sn}", debug)

        sn_in_community = False
        for comm in initial_communities:
            # 8-9. Find the community to which sn belongs to
            if sn in comm:
                # 10. Insert v into comm and remove it from nodes_degrees
                comm.append(v)
                del nodes_degrees[v]
                sn_in_community = True
                break

        # 4. If sn is not in any community
        if not sn_in_community:
            # 5. Create a new community and assign v and sn to it
            community = [v, sn]
            # 6. Insert the new community into community structure
            initial_communities.append(community)
            # 7. Remove v and sn from NS as they are classified
            del nodes_degrees[v]
            del nodes_degrees[sn]

        _printd(f"Coms. in comp. created so far: {initial_communities}", debug)

    _printd(f"Coms. in comp. created: {initial_communities}\n\n", debug)
    return initial_communities


def _consolide_communities_in_component(
    initial_communities: list[list[Any]],
    G: nx.Graph,
    delta: float,
    debug: bool,
    weight_attr: Optional[str] = None,
) -> list[list[Any]]:
    """
    Phase 2 - Community consolidation.

    :param initial_communities: initial divison of nodes into communities
    :param G: input graph
    :param delta: merging threshold
    :param weight_attr: edge attribute to take weight from, defaults to None

    :return: final divison of nodes into communities
    """
    # pylint: disable=R0914
    assert 0 <= delta <= 1, "Delta must be in range [0, 1]"

    # 12. Initialise Final Communities
    final_communities = initial_communities.copy()

    # MCz: edge case - stop it there is only one community left
    while len(final_communities) > 1:

        # 13-14. Calculate merging index (ψi) for each community
        merging_idx_list = np.array(
            [
                _merging_index(G=G, community=comm, weight_attr=weight_attr)
                for comm in final_communities
            ]
        )
        _printd(f"Community merging indices: {merging_idx_list}", debug)

        # 15. Select the community with the lowest merging index (Cx)
        lowest_merging_idx = _get_extreme_val(merging_idx_list, np.argmin)
        _printd(
            "Lowest merging index of community "
            + f"{lowest_merging_idx[0]}: {lowest_merging_idx[1]}",
            debug,
        )

        # 19. Stop community consolidation if 'ψi' > 'δ'
        if lowest_merging_idx[1] > delta:
            break

        # 16. Find the most similar community (Cy) to (Cx) and merge the two
        # communites to form a new community (Cn)
        similarity_list = np.ones(len(final_communities)) * -1
        for idx, comm in enumerate(final_communities):
            if idx == lowest_merging_idx[0]:
                continue
            dsc_sum = 0.0
            for u in final_communities[lowest_merging_idx[0]]:
                for v in comm:
                    dsc_sum += _dsc(G=G, u=u, v=v)
            similarity_list[idx] = dsc_sum / len(
                final_communities[lowest_merging_idx[0]]
            )
        _printd(
            "Communities similarity to community "
            + f"{lowest_merging_idx[0]}: {similarity_list}",
            debug,
        )
        highest_similarity = _get_extreme_val(similarity_list, np.argmax)
        _printd(
            "Most similar community "
            + f"{highest_similarity[0]}: {highest_similarity[1]}",
            debug,
        )
        new_community = (
            final_communities[lowest_merging_idx[0]]
            + final_communities[highest_similarity[0]]
        )
        _printd(f"New community: {new_community}", debug)

        # 17. Calculate the merging index (ψn) for new community (Cn)
        merging_idx = _merging_index(G, new_community, weight_attr)
        _printd(f"New community merging index: {merging_idx}", debug)

        # 18. Replace two communites 'Cx' and 'Cy' with new community 'Cn' in
        # the final community set (FC)
        _final_communities = []
        for idx, comm in enumerate(final_communities):
            if idx == lowest_merging_idx[0]:
                _final_communities.append(new_community)
            elif idx == highest_similarity[0]:
                continue
            else:
                _final_communities.append(comm)
        final_communities = _final_communities
        _printd(f"Final coms. in comp. so far: {final_communities}\n\n", debug)

    # 20. Return Final Communities
    _printd(f"Final Coms. in component: {final_communities}\n\n", debug)
    return final_communities


def detect_communities(
    net: nx.Graph,
    merging_idx_threshold: float,
    debug: bool = False,
    weight_attr: Optional[str] = None,
) -> list[list[Any]]:
    """
    Detect communities according to the CBIM algorithm.

    This is an implementation of the Algorithm 2 from "CBIM: Community-based
    influence maximization in multilayer networks" by Venkatakrishna Rao,
    C. Ravindranath Chowdary (https://doi.org/10.1016/j.ins.2022.07.103) This
    implementation is based on https://github.com/doublejv/CBIM-Implementation,
    but with several bugs fixed.

    :param net: input graph
    :param merging_idx_threshold: a threshold above which communities will not
        be merged anymore during consolidation phase
    :param debug: if print debug statements, defaults to False
    :param weight_attr: which (if any) attribute of edges to use as a weight,
        defaults to None

    :return: divison of the nodes into disjoint communities
    """
    if isinstance(net, nx.DiGraph):
        components = [
            nx.subgraph(net, component)
            for component in nx.weakly_connected_components(net)
        ]
    elif isinstance(net, nx.Graph):
        components = [
            nx.subgraph(net, component)
            for component in nx.connected_components(net)
        ]
    else:
        raise ValueError(f"Graph type {type(net)} is not supported!")
    _printd(f"Found {len(components)} components\n\n", debug)

    graph_communities = []
    for component in components:
        # steps 1-11 of the Algorithm 2
        initial_communities = _initialise_communities_in_component(
            G=component, debug=debug, weight_attr=weight_attr
        )
        # steps 12-20 of the Algorithm 2
        final_communities = _consolide_communities_in_component(
            G=component,
            initial_communities=initial_communities,
            debug=debug,
            delta=merging_idx_threshold,
            weight_attr=weight_attr,
        )
        graph_communities.extend(final_communities)
    _printd(f"Final Communities in graph: {graph_communities}\n\n", debug)
    return graph_communities


def _compute_katz_centralities(
    G: nx.Graph,
    communities: list[list[Any]],
    weight_attr: Optional[str] = None,
) -> list[list[dict[str, Any]]]:
    """Calculate Katz centrality for each node in each community."""
    katz_centralities = []
    for community in communities:
        com_net = G.subgraph(community)
        try:
            kc_raw = nx.katz_centrality(
                com_net,
                alpha=0.1,
                beta=1.0,
                max_iter=10000,
                weight=weight_attr,
            )
            katz_centralities.append(
                [
                    {"node_id": k, "katz_centrality": v}
                    for k, v in kc_raw.items()
                ]
            )
        except PowerIterationFailedConvergence:
            katz_centralities.append(
                [
                    {"node_id": n, "katz_centrality": 0.0}
                    for n in com_net.nodes()
                ]
            )
            warnings.warn("Katz centrality computation failed!", stacklevel=1)
    return katz_centralities


def _select_seeds_from_katz(
    katz_centralities: list[list[dict[str, Any]]], quotas: list[int]
) -> set[Any]:
    """Select top-k nodes with highest katz-centrality from each community."""
    seeds_ranked = []
    for quota, decomposed_community in zip(quotas, katz_centralities):
        # sort all the nodes in each community based on Katz Centrality
        # Coeficient in descending order
        df = (
            pd.DataFrame(decomposed_community)
            .sort_values(["katz_centrality"], ascending=[False])
            .reset_index(drop=True)
        )
        # select the quota number of highest Katz centrality coeficient nodes
        # as seed nodes from each community (Ci)
        seeds_ranked.extend(df.iloc[:quota]["node_id"].to_list())
    return set(seeds_ranked)


def cbim_seed_selection(
    net: nx.Graph,
    num_seeds: int,
    merging_idx_threshold: float,
    debug: bool = False,
    weight_attr: Optional[str] = None,
) -> set[Any]:
    """
    Select <num_seeds> from <G> according to CBIM algorithm.

    The routine was published by Venkatakrishna Rao, C. Ravindranath Chowdary
    in CBIM: Community-based influence maximization in multilayer networks"
    (https://doi.org/10.1016/j.ins.2022.07.103) which was published in
    "Information Sciences", 2022, Volume 609. This implementation is based on
    code from here https://github.com/doublejv/CBIM-Implementation, but with
    several bugs fixed.

    :param net: input graph
    :param merging_idx_threshold: a threshold above which communities will not
        be merged anymore during consolidation phase
    :param num_seeds: number of seeds to pick, in form of number of nodes
    :param debug: if print debug statements, defaults to False
    :param weight_attr: which (if any) attribute of edges to use as a weight,
        defaults to None

    :return: seed set according to the given budget
    """
    communities = detect_communities(
        net=net,
        merging_idx_threshold=merging_idx_threshold,
        debug=debug,
        weight_attr=weight_attr,
    )
    # steps 2-7 of the Algorithm 3
    k_cenrt = _compute_katz_centralities(
        communities=communities, G=net, weight_attr=weight_attr
    )
    # steps 8-9 of the Algorithm 3
    quotas = compute_seed_quotas(
        G=net, communities=communities, num_seeds=num_seeds
    )
    # steps 10-11 of the Algorithm 3
    return _select_seeds_from_katz(katz_centralities=k_cenrt, quotas=quotas)


def cbim_seed_ranking(
    net: nx.Graph,
    merging_idx_threshold: float,
    debug: bool = False,
    weight_attr: Optional[str] = None,
) -> list[Any]:
    """
    Rank all nodes from <net> according to CBIM algorithm.

    The routine is a modification of function cbim_seed_selection so that not a
    given fraction of most influential nodes is returned, but all of them.

    :param net: input graph
    :param merging_idx_threshold: a threshold above which communities will not
        be merged anymore during consolidation phase
    :param num_seeds: number of seeds to pick, in form of number of nodes
    :param debug: if print debug statements, defaults to False
    :param weight_attr: which (if any) attribute of edges to use as a weight,
        defaults to None

    :return: an ordered ranking of nodes (most influential are at the beggining
        of the list)
    """
    communities = detect_communities(
        net=net,
        merging_idx_threshold=merging_idx_threshold,
        debug=debug,
        weight_attr=weight_attr,
    )
    k_cenrt = _compute_katz_centralities(net, communities, weight_attr)

    # now, basing on communities build the ranking iteratively
    ranking = []
    for i in range(1, len(net.nodes) + 1):
        quotas = compute_seed_quotas(net, communities, i)
        seeds = _select_seeds_from_katz(k_cenrt, quotas)
        for seed in seeds:
            if seed not in ranking:
                ranking.append(seed)
    return ranking
