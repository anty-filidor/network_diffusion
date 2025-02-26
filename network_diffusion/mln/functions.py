# Copyright (c) 2023 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Script with some functions of NetworkX extended to multilayer networks."""

import warnings
from typing import Iterator

import matplotlib.pyplot as plt
import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def all_neighbours(
    net: MultilayerNetwork, actor: MLNetworkActor
) -> Iterator[MLNetworkActor]:
    """
    Return all of the neighbours of an actor in the graph.

    If the graph is directed returns predecessors as well as successors.
    Overloads networkx.classes.functions.all_neighbours.
    """
    neighbours = []
    for l_name in actor.layers:
        l_graph = net.layers[l_name]
        neighbours.extend([*nx.all_neighbors(l_graph, actor.actor_id)])
    return iter([net.get_actor(i) for i in set(neighbours)])


def squeeze_by_neighbourhood(
    net: MultilayerNetwork, preserve_mlnetworkactor_objects: bool = True
) -> nx.Graph:
    """
    Squeeze multilayer network to a single layer `nx.Graph`.

    All actors are preserved, links are produced according to naighbourhood
    between actors regardless layers.

    :param net: a multilayer network to be squeezed
    :param preserve_mlnetworkactor_objects: a flag indicating how to represent
        nodes - by `MLNetworkActor` or primitive types like `str`, `int` etc.
    :return: a networkx.Graph representing `net`
    """
    squeezed_net = nx.Graph()
    for actor in net.get_actors():
        for neighbour in all_neighbours(net, actor):
            edge = (
                (actor, neighbour)
                if preserve_mlnetworkactor_objects
                else (actor.actor_id, neighbour.actor_id)
            )
            squeezed_net.add_edge(*edge)
        squeezed_net.add_node(
            actor if preserve_mlnetworkactor_objects else actor.actor_id
        )
    assert net.get_actors_num() == len(squeezed_net)
    return squeezed_net


def draw_mln(net: MultilayerNetwork, dpi: int = 300) -> None:
    """Draw briefly a given multilayer network."""
    if net.get_actors_num() > 100:
        warnings.warn(
            f"Too large network ({net.get_actors_num()}). \
                Visualisation can be crippled.",
            stacklevel=1,
        )
    pos = nx.drawing.shell_layout([a.actor_id for a in net.get_actors()])
    fig, axs = plt.subplots(nrows=1, ncols=len(net.layers))
    fig.set_dpi(dpi)
    for idx, (layer_name, layer_graph) in enumerate(net.layers.items()):
        axs[idx].set_title(layer_name)
        nx.draw(layer_graph, ax=axs[idx], pos=pos)
        nx.drawing.draw_networkx_labels(layer_graph, ax=axs[idx], pos=pos)
    plt.show()


def remove_selfloop_edges(net: MultilayerNetwork) -> MultilayerNetwork:
    """Remove selfloop edges from the network."""
    for l_name in net.layers:
        selfloop_edges = list(nx.selfloop_edges(net.layers[l_name]))
        net.layers[l_name].remove_edges_from(selfloop_edges)

    return net
