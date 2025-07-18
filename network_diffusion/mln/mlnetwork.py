# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A script where a multilayer network is defined."""

import random
import warnings
from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np
from uunet import multinet

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class MultilayerNetwork:
    """
    A basic container for the multilayer network.

    Multilayer network is represented as a dictionary of `networkx` graphs.
    """

    def __init__(self, layers: dict[str, nx.Graph]) -> None:
        """
        Create an object.

        Please note, that all nodes will have initialised "status" attribute to
        make the network ready for simulations of spreading phenomena.

        :param layers: a layers as `networkx` graphs.
        """
        prepared_layers = self._prepare_nodes_attribute(layers)
        self.layers = prepared_layers

    @classmethod
    def from_mpx(cls, file_path: str) -> "MultilayerNetwork":
        """
        Load multilayer network from a `mpx` file.

        `mpx` is a standard provided by `multinet` library mainatined by
        Uppsala University Information Laboratory: https://github.com/uuinfolab
        Pleas note that `multinet` is mostly written in C/C++ and it is not
        possible to control its random numger generator.

        :param file_path: path to the file
        """
        uu_net = multinet.read(file_path)
        nx_net = multinet.to_nx_dict(uu_net)
        return cls(nx_net)

    @classmethod
    def from_nx_layers(
        cls,
        network_list: list[nx.Graph],
        layer_names: list[Any] | None = None,
    ) -> "MultilayerNetwork":
        """
        Load multilayer network as list of layers and list of its labels.

        :param network_list: list of nx networks
        :param layer_names: list of layer names. It can be none, then labels
            are set automatically
        """
        layers = {}
        if layer_names is not None:
            assert len(network_list) == len(
                layer_names
            ), "Length of network list and metadata list is not equal"
            for layer, name in zip(network_list, layer_names):
                layers.update({name: layer})
        else:
            for i, layer in enumerate(network_list, start=0):
                name = f"layer_{i}"
                layers.update({name: layer})

        return cls(layers)

    @classmethod
    def from_nx_layer(
        cls, network_layer: nx.Graph, layer_names: list[Any]
    ) -> "MultilayerNetwork":
        """
        Create multiplex network from one nx.Graph layer and layers names.

        Note that `network_layer` is replicated through all layers.

        :param network_layer: basic layer which is replicated through all ones
        :param layer_names: names for layers in multiplex network
        """
        layers = {}
        for name in layer_names:
            layers.update({name: deepcopy(network_layer)})

        return cls(layers)

    @staticmethod
    def _prepare_nodes_attribute(
        layers: dict[str, nx.Graph]
    ) -> dict[str, nx.Graph]:
        """Prepare network for the experiment."""
        for layer in layers.values():
            status_dict = {n: None for n in layer.nodes()}
            nx.set_node_attributes(layer, status_dict, "status")
        return layers

    def __getitem__(self, key: str) -> nx.Graph:
        """Get 'key' layer of the network."""
        return self.layers[key]

    def __copy__(self) -> "MultilayerNetwork":
        """Create a copy of the network."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> "MultilayerNetwork":
        """Create a deep copy of the network."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, val in self.__dict__.items():
            setattr(result, key, deepcopy(val, memo))
        return result

    def __len__(self) -> int:
        """Return length of the network, i.e. num of actors."""
        return self.get_actors_num()

    def __str__(self) -> str:
        """Print out quickly parameters of the network."""
        return self._get_description_str()

    def _get_description_str(self) -> str:
        """
        Get string describing the object.

        :param full_graph: a flag, if true method prints out all propagation
            model states with those with 0 weight

        :return: string describing object
        """
        assert len(self.layers) > 0, "Import network to the object first!"
        final_str = f"{BOLD_UNDERLINE}\nnetwork parameters\n{THIN_UNDERLINE}\n"
        final_str += "general parameters:\n"
        final_str += f"\tnumber of layers - {len(self.layers)}\n"
        final_str += f"\tnumber of actors - {self.get_actors_num()}\n"

        edges_nb, nodes_nb = [], []
        for layer_graph in self.layers.values():
            edges_nb.append(len(layer_graph.edges()))
            nodes_nb.append(len(layer_graph.nodes()))
        final_str += f"\tnumber of nodes - {sum(nodes_nb)}\n"
        final_str += f"\tnumber of edges - {sum(edges_nb)}\n"

        for name, graph in self.layers.items():
            final_str += f"\nlayer '{name}' parameters:\n"
            final_str += (
                f"\tgraph type - {type(graph)}\n\tnumber of nodes - "
                f"{len(graph.nodes())}\n\tnumber of edges - "
                f"{len(graph.edges())}\n"
            )
            avg_deg = np.average([_[1] for _ in graph.degree])
            final_str += f"\taverage degree - {round(avg_deg, 4)}\n"
            if len(graph) > 0:
                final_str += (
                    f"\tclustering coefficient - "
                    f"{round(nx.average_clustering(graph), 4)}\n"
                )
            else:
                final_str += "\tclustering coefficient - nan\n"
        final_str += BOLD_UNDERLINE

        return final_str

    def is_directed(self) -> bool:
        """Check whether at least one layer is a DirectedGraph."""
        return bool(
            np.array(
                [l_graph.is_directed() for l_graph in self.layers.values()]
            ).any()
        )

    def copy(self) -> "MultilayerNetwork":
        """Create a deep copy of the network."""
        copied_instance = MultilayerNetwork(
            {name: deepcopy(graph) for name, graph in self.layers.items()}
        )
        return copied_instance

    def subgraph(self, actors: list[MLNetworkActor]) -> "MultilayerNetwork":
        """
        Return a subgraph of the network.

        The induced subgraph of the graph contains the nodes in `nodes` and the
        edges between those nodes. This is an equivalent of nx.Graph.subgraph.
        """
        nodes_to_keep: dict[str, list[Any]] = {
            l_name: [] for l_name in self.get_layer_names()
        }
        for actor in actors:
            for l_name in actor.layers:
                nodes_to_keep[l_name].append(actor.actor_id)

        sub_layers = {}
        for l_name, kept_nodes in nodes_to_keep.items():
            l_subgraph = self.layers[l_name].subgraph(kept_nodes).copy()
            sub_layers[l_name] = l_subgraph

        return MultilayerNetwork(sub_layers)

    def is_multiplex(self) -> bool:
        """Check if network is multiplex."""
        actors = {a.actor_id for a in self.get_actors()}
        for layer in self.layers:
            if len(actors.difference(self[layer])) > 0:
                return False
        return True

    def to_multiplex(self) -> tuple["MultilayerNetwork", dict[str, set[Any]]]:
        """Convert network to multiplex one by adding missing nodes."""
        if self.is_multiplex():
            warnings.warn("Network is already multiplex!", stacklevel=1)
            return self.copy(), {}

        actors = {a.actor_id for a in self.get_actors()}
        multiplexed_layers = {}
        added_nodes = {}
        for layer in self.layers:
            missing_nodes = actors.difference(self[layer])
            updated_layer = self[layer].copy(as_view=False)
            updated_layer.add_nodes_from(missing_nodes)
            multiplexed_layers[layer] = updated_layer
            added_nodes[layer] = missing_nodes
        return MultilayerNetwork(multiplexed_layers), added_nodes

    def get_actors(self, shuffle: bool = False) -> list[MLNetworkActor]:
        """
        Get actors that exist in the network and read their states.

        :param shuffle: a flag that determines whether to shuffle actor list
        :return: a list with actors that live in the network
        """
        actor_dict: dict[str, dict] = {}
        for layer_name, layer_graph in self.layers.items():
            for node in layer_graph.nodes:
                if node not in actor_dict:
                    actor_dict[node] = {}
                actor_dict[node][layer_name] = layer_graph.nodes[node][
                    "status"
                ]

        actor_list = [
            MLNetworkActor(actor_id=a_name, layers_states=a_layers_states)
            for a_name, a_layers_states in actor_dict.items()
        ]
        if shuffle:
            random.shuffle(actor_list)

        return actor_list

    def get_actor(self, actor_id: Any) -> MLNetworkActor:
        """Get actor data basing on its name."""
        layers_states = {}
        for layer_name, layer_graph in self.layers.items():
            if not layer_graph.nodes.get(actor_id):
                continue
            layers_states[layer_name] = layer_graph.nodes[actor_id]["status"]
        if len(layers_states) == 0:
            raise ValueError(
                f"Actor with id: {actor_id} doesn't exist in the network!"
            )
        return MLNetworkActor(actor_id=actor_id, layers_states=layers_states)

    def get_actors_num(self) -> int:
        """Get number of actors that live in the network."""
        actors_set: set[Any] = set()
        for layer_graph in self.layers.values():
            actors_set = actors_set.union(set(layer_graph.nodes))
        return len(actors_set)

    def get_nodes_num(self) -> dict[str, int]:
        """Get number of nodes that live in each layer of the network."""
        nodes_num = {}
        for layer_name, layer_graph in self.layers.items():
            nodes_num[layer_name] = len(layer_graph.nodes)
        return nodes_num

    def get_links(
        self, actor_id: Any | None = None
    ) -> set[tuple[MLNetworkActor, MLNetworkActor]]:
        """
        Get links connecting all actors from the network regardless layers.

        :return: a set with edges between actors
        """
        if actor_id is not None:
            actor = self.get_actor(actor_id=actor_id)
            actor_edges_per_layer = [
                [
                    (self.get_actor(n_1), self.get_actor(n_2))
                    for n_1, n_2 in self.layers[l_name].edges(actor_id)
                ]
                for l_name in actor.layers
            ]
        else:
            actor_edges_per_layer = [
                [
                    (self.get_actor(n_1), self.get_actor(n_2))
                    for n_1, n_2 in l_graph.edges()
                ]
                for l_graph in self.layers.values()
            ]

        actor_edges_merged: set[tuple[MLNetworkActor, MLNetworkActor]] = set()
        for l_edges in actor_edges_per_layer:
            actor_edges_merged = actor_edges_merged.union(l_edges)
        return actor_edges_merged

    def get_layer_names(self) -> list[str]:
        """
        Get names of layers in the network.

        :return: list of layers' names
        """
        return [*self.layers.keys()]
