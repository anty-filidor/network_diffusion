"""Script with functions of NetworkX extended to multilayer networks."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def degree(net: MultilayerNetwork) -> Dict[MLNetworkActor, int]:
    """Return number of connecting links per all actors from the network."""
    degrees: Dict[MLNetworkActor, int] = {}
    for actor in net.get_actors():
        a_neighbours = 0
        for l_name in actor.layers:
            a_neighbours += len(net.layers[l_name].adj[actor.actor_id])
        degrees[actor] = a_neighbours
    return degrees


def neighbourhood_size(net: MultilayerNetwork) -> Dict[MLNetworkActor, int]:
    """Return naighbourhood sizes of all actors from the network."""
    neighbourhood_sizes: Dict[MLNetworkActor, int] = {}
    for actor in net.get_actors():
        a_neighbours: Set[Any] = set()
        for l_name in actor.layers:
            a_neighbours = a_neighbours.union(
                set(net.layers[l_name].adj[actor.actor_id].keys())
            )
        neighbourhood_sizes[actor] = len(a_neighbours)
    return neighbourhood_sizes


def multiplexing_coefficient(net: MultilayerNetwork) -> float:
    """
    Compute multiplexing coefficient.

    Multiplexing coefficient is defined as proportion of number of nodes
    common to all layers to number of all unique nodes in entire network

    :return: (float) multiplexing coefficient
    """
    assert len(net.layers) > 0, "Import network to the object first!"

    # create set of all nodes in the network
    all_nodes = set()
    for graph in net.layers.values():
        all_nodes.update([*graph.nodes()])

    # create set of all common nodes in the network over the layers
    common_nodes = set()
    for node in all_nodes:
        _ = True
        for layer in net.layers.values():
            if node not in layer:
                _ = False
                break
        if _ is True:
            common_nodes.add(node)

    if len(all_nodes) > 0:
        return len(common_nodes) / len(all_nodes)
    else:
        return 0


def number_of_selfloops(net: MultilayerNetwork) -> int:
    """
    Return the number of selfloop edges in the entire network.

    A selfloop edge has the same node at both ends. Overloads networkx.classes.
    functions.number_of_selfloops.
    """
    selfloops_num = 0
    for l_graph in net.layers.values():
        selfloops_num += sum(1 for _ in nx.selfloop_edges(l_graph))
    return selfloops_num


def all_neighbors(
    net: MultilayerNetwork, actor: MLNetworkActor
) -> Iterator[MLNetworkActor]:
    """
    Return all of the neighbors of an actor in the graph.

    If the graph is directed returns predecessors as well as successors.
    Overloads networkx.classes.functions.all_neighbours.
    """
    neighbours = []
    for l_name in actor.layers:
        l_graph = net.layers[l_name]
        neighbours.extend([*nx.all_neighbors(l_graph, actor.actor_id)])
    return iter([net.get_actor(i) for i in set(neighbours)])


def core_number(net: MultilayerNetwork) -> Dict[MLNetworkActor, int]:
    """
    Return the core number for each actor.

    A k-core is a maximal subgraph that contains actors of degree k or more. A
    core number of a node is the largest value k of a k-core containing that
    node. Not implemented for graphs with parallel edges or self loops.
    Overloads networkx.algorithms.core.core_number.

    :param net: multilayer network
    :return: dictionary keyed by actor to the core number.
    """
    # pylint: disable=C0103

    if number_of_selfloops(net) > 0:
        raise ValueError("Input graph has self loops which is not permitted!")
    degrees_like = neighbourhood_size(net)

    # Sort nodes by degree.
    nodes = sorted(degrees_like, key=lambda x: degrees_like[x])
    bin_boundaries = [0]
    curr_degree = 0
    for i, v in enumerate(nodes):
        if degrees_like[v] > curr_degree:
            bin_boundaries.extend([i] * (degrees_like[v] - curr_degree))
            curr_degree = degrees_like[v]
    node_pos = {v: pos for pos, v in enumerate(nodes)}

    # The initial guess for the core number of a node is its degree.
    core = degrees_like
    nbrs = {v: list(all_neighbors(net, v)) for v in net.get_actors()}
    for v in nodes:
        for u in nbrs[v]:
            if core[u] > core[v]:
                nbrs[u].remove(v)
                pos = node_pos[u]
                bin_start = bin_boundaries[core[u]]
                node_pos[u] = bin_start
                node_pos[nodes[bin_start]] = pos
                nodes[bin_start], nodes[pos] = nodes[pos], nodes[bin_start]
                bin_boundaries[core[u]] += 1
                core[u] -= 1
    return core


def _core_subgraph(
    net: MultilayerNetwork,
    k_filter: Callable,
    k: Optional[int] = None,
    core: Optional[Dict[MLNetworkActor, int]] = None,
) -> MultilayerNetwork:
    """
    Return the subgraph induced by nodes passing filter `k_filter`.

    Overloads networkx.algorithms.core._core_subgraph.

    :param net: The graph or directed graph to process
    :param k_filter: This function filters the nodes chosen. It takes three
        inputs: A node of net, the filter's cutoff, and the core dict of the
        graph. The function should return a Boolean value.
    :param k: The order of the core. If not specified use the max core number.
        This value is used as the cutoff for the filter.
    :param core: Precomputed core numbers keyed by node for the graph `net`. If
        not specified, the core numbers will be computed from `net`.

    :return: core subgraph
    """
    if core is None:
        core = core_number(net)
    if k is None:
        k = max(core.values())
    nodes = list(v for v in core if k_filter(v, k, core))
    return net.subgraph(nodes)


def k_shell_mln(
    net: MultilayerNetwork,
    k: Optional[int] = None,
    core_number: Optional[Dict[MLNetworkActor, int]] = None,
) -> MultilayerNetwork:
    """
    Return the k-shell of net with degree computed actorwise.

    The k-shell is the subgraph induced by actors with core number k. That is,
    actors in the k-core that are not in the (k+1)-core. The k-shell is not
    implemented for graphs with self loops or parallel edges. Overloads
    networkx.algorithms.core.k_shell.

    :param net: A graph or directed graph.
    :param k: The order of the shell. If not specified return the outer shell.
    :param core_number: Precomputed core numbers keyed by node for the graph
        `net`. If not specified, the core numbers will be computed from `net`.
    :return: The k-shell subgraph
    """
    # pylint: disable=W0621
    return _core_subgraph(net, lambda v, k, c: c[v] == k, k, core_number)


def voterank_actorwise(
    net: MultilayerNetwork, number_of_actors: Optional[int] = None
) -> List[MLNetworkActor]:
    """
    Select a list of influential ACTORS in a graph using VoteRank algorithm.

    VoteRank computes a ranking of the actors in a graph based on a voting
    scheme. With VoteRank, all actors vote for each of its neighbours and the
    actor with the highest votes is elected iteratively. The voting ability of
    neighbors of elected actors is decreased in subsequent turns. Overloads
    networkx.algorithms.core.k_shell.

    :param net: multilayer network
    :param number_of_actors: number of ranked actors to extract (default all).

    :return: ordered list of computed seeds, only actors with positive number
        of votes are returned.
    """
    if net.is_directed():
        raise NotImplementedError(
            "Voterank for directed networks is not implemented!"
        )

    influential_actors: List[MLNetworkActor] = []
    if len(net) == 0:
        return influential_actors
    if number_of_actors is None or number_of_actors > len(net):
        number_of_actors = len(net)
    vote_rank = {}

    # compute average neighbourhood size
    agv_nbs = sum(deg for _, deg in neighbourhood_size(net=net).items()) / len(
        net
    )

    # step 1 - initiate all nodes to (0,1) (score, voting ability)
    for actor in net.get_actors():
        vote_rank[actor] = [0.0, 1.0]

    # Repeat steps 1b to 4 until num_seeds are elected.
    for _ in range(number_of_actors):

        # step 1b - reset rank
        for actor in net.get_actors():
            vote_rank[actor][0] = 0

        # step 2 - vote
        for actor, nbr in net.get_links():
            vote_rank[actor][0] += vote_rank[nbr][1]
            vote_rank[nbr][0] += vote_rank[actor][1]
        for actor in influential_actors:
            vote_rank[actor][0] = 0

        # step 3 - select top actor
        top_actor = max(net.get_actors(), key=lambda x: vote_rank[x][0])
        if vote_rank[top_actor][0] == 0:
            return influential_actors
        influential_actors.append(top_actor)
        # weaken the selected actor
        vote_rank[top_actor] = [0, 0]

        # step 4 - update voterank properties
        for _, nbr in net.get_links(top_actor.actor_id):
            vote_rank[nbr][1] -= 1 / agv_nbs
            vote_rank[nbr][1] = max(vote_rank[nbr][1], 0)

    return influential_actors


def squeeze_by_neighbourhood(net: MultilayerNetwork) -> nx.Graph:
    """
    Squeeze multilayer network to single layer by neighbourhood of actors.

    All actors are preserved, links are produced according to naighbourhood
    between actors regardless layers.

    :param net: a multilayer network to be squeezed
    :return: a networkx.Graph representing `net`
    """
    squeezed_net = nx.Graph()
    for actor in net.get_actors():
        for neighbour in all_neighbors(net, actor):
            squeezed_net.add_edge(actor, neighbour)
        squeezed_net.add_node(actor)
    assert net.get_actors_num() == len(squeezed_net)
    return squeezed_net
