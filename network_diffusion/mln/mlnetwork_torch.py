# Copyright (c) 2024 by Michał Czuba.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A converter from `MultilayerNetwork` to the sparse repr. in PyTorch."""

from dataclasses import dataclass
from typing import Any

import networkx as nx
import torch
import torch_geometric as pyg
from bidict import bidict

from network_diffusion.mln.mlnetwork import MultilayerNetwork


def _prepare_mln_for_conversion(
    net: MultilayerNetwork,
) -> tuple[MultilayerNetwork, bidict, dict[str, set[Any]] | None]:
    """
    Prepare `MultilayerNetwork` for conversion to torch representation.

    If network is not multiplex, then multiplicity actoss all layers will be
    imposed. Names of the actors will be converted to integers.

    :param net: a copy of the multilayer network prepared for conversion
    :return: a new instance of `MultilayerNetwork` prepared for conversion,
        a bi-directional map of the old and new actors' names, a dict of nodes'
        sets added to make the network multiplex (with their original ids)
    """
    if not net.is_multiplex():
        net_m, added_nodes = net.to_multiplex()
    else:
        net_m, added_nodes = net, None
    ac_map = {ac.actor_id: idx for idx, ac in enumerate(net_m.get_actors())}
    l_dict = {
        l_name: nx.relabel_nodes(l_graph, mapping=ac_map, copy=True)
        for l_name, l_graph in net_m.layers.items()
    }
    return MultilayerNetwork(l_dict), bidict(ac_map), added_nodes


def _mln_to_sparse(
    net: MultilayerNetwork, actor_order: list[Any]
) -> tuple[torch.Tensor, list[str]]:
    """
    Converse `MultilayerNetwork` to an adjacency matrix as a tensor.

    :param net: `MultilayerNetwork` to be converted, must be multiplex and have
        actors' ids represented as integers
    :param actor_order: order of actors' ids to be used in the output adjacency
        tensor
    :return: an adjacency matrix as a sparse tensor and a list of layer names
        ordered as in input adjacency matrix
    """
    adj, layers = [], []
    for l_name, l_graph in net.layers.items():
        lg_idx, lg_val = pyg.utils.from_scipy_sparse_matrix(
            nx.adjacency_matrix(G=l_graph, nodelist=actor_order, weight=None)
        )
        lg_adj = torch.sparse_coo_tensor(
            indices=lg_idx,
            values=lg_val,
            size=[len(actor_order)] * 2,
            is_coalesced=True,
            check_invariants=True,
        )
        adj.append(lg_adj)
        layers.append(l_name)
    return torch.stack(adj).coalesce(), layers


def _create_nodes_mask(
    layers_order: list[str],
    actors_map: bidict,
    nodes_added: dict[str, set[Any]] | None,
    debug: bool = False,
) -> torch.Tensor:
    """
    Create mask which marks nodes that were added artifically (as ones).

    :param layers_order: names of layers in the order that is preserved in the
        sparse adjacency tensor
    :param actors_map: map of actor names `Any` -> `int` between the original
        network and its sparse representation
    :param nodes_added: a dict of sets of nodes added to make the net multiplex
        (with the original ids)
    :param debug: a flag whether to print debug info
    :return: tensor of shape **[nb. layers x nb. actors]**
    """
    n_mask = torch.zeros([len(layers_order), len(actors_map)])
    if not nodes_added:
        return n_mask
    for l_name, l_added_nodes in nodes_added.items():
        if debug:
            print(f"layer: {l_name}, added nodes: {l_added_nodes}")
        for l_added_node in l_added_nodes:
            if debug:
                print(f"map: {l_added_node}->{actors_map[l_added_node]}")
            n_mask[layers_order.index(l_name), actors_map[l_added_node]] = 1.0
    return n_mask


@dataclass(frozen=True)
class MultilayerNetworkTorch:
    """
    Representation of `MultilayerNetwork` in a tensor notation.

    Note, that in order to provide consistency between channels of an adjacency
    matrix, the network is converted to multiplex with nodes added artifically
    marked in the property `nodes_mask`

    :param adjacency_tensor: adjacency matrix as a sparse tensor shaped as
        **[nb. layers x nb. actors x nb. actors]**
    :param layers_order: names of layers in an order that is preserved in
        `adjacency_tensor`
    :param actors_map: map of actor names `Any` -> `int` between the original
        network and its sparse representation
    :param nodes_mask: mask of nodes added while making the network multiplex
        ordered in the same way as the nodes in `adjacency_tensor`
    """

    adjacency_tensor: torch.Tensor
    layers_order: list[str]
    actors_map: bidict
    nodes_mask: torch.Tensor

    @classmethod
    def from_mln(cls, net: MultilayerNetwork) -> "MultilayerNetworkTorch":
        """Represent net in a tensor notation."""
        net_converted, ac_map, nodes_added = _prepare_mln_for_conversion(net)
        adj, l_order = _mln_to_sparse(net_converted, list(ac_map.values()))
        n_mask = _create_nodes_mask(l_order, ac_map, nodes_added)
        return cls(adj, l_order, ac_map, n_mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} at {id(self)}\n"
            f"adjacency_tensor: {self.adjacency_tensor}\n"
            f"layers_order: {self.layers_order}\n"
            f"actors map: {self.actors_map}\n"
            f"nodes_mask: {self.nodes_mask}\n"
        )
