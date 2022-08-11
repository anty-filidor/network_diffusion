# Copyright 2020 by Michał Czuba, Piotr Bródka. All Rights Reserved.
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

"""Functions for the multilayer network definition."""

# pylint: disable=W0141

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from network_diffusion.utils import read_mlx


class MultilayerNetwork:
    """Container for multilayer network."""

    def __init__(self) -> None:
        """Create empty object."""
        self.layers: Dict[str, nx.Graph] = {}

    def load_mlx(self, file_path: str) -> None:
        """
        Load multilayer network from mlx file.

        Note, that is omits some non-important attributes of network defined in
        the file, i.e. node attributes.

        :param file_path: path to the file
        """
        # clear former layers of the object
        self.layers = {}

        # read mlx file
        raw_data = read_mlx(file_path)

        # create layers
        if "layers" in raw_data:
            for layer_name, layer_type in raw_data["layers"]:
                if "DIRECTED" in layer_type:
                    self.layers[layer_name] = nx.DiGraph()
                else:
                    print("unrecognised layer")
        else:
            raise ResourceWarning("file corrupted - no layers defined")

        # import edges
        if "edges" in raw_data:
            for edge in raw_data["edges"]:
                # if edge definition is not corrupted go into it
                if len(edge) >= 3 and edge[2] in self.layers:
                    self.layers[edge[2]].add_edge(edge[0], edge[1])

        self._prepare_nodes_attribute()

    def load_layers_nx(
        self,
        network_list: List[nx.Graph],
        layer_names: Optional[List[Any]] = None,
    ) -> None:
        """
        Load multilayer network as list of layers and list of its labels.

        :param network_list: list of nx networks
        :param layer_names: list of layer names. It can be none, then labels
            are set automatically
        """
        # clear former layers of the object
        self.layers = {}

        if layer_names is not None:
            assert len(network_list) == len(
                layer_names
            ), "Length of network list and metadata list is not equal"
            for layer, name in zip(network_list, layer_names):
                self.layers.update({name: layer})
        else:
            for i, layer in enumerate(network_list, start=0):
                name = f"layer_{i}"
                self.layers.update({name: layer})

        self._prepare_nodes_attribute()

    def load_layer_nx(
        self, network_layer: nx.Graph, layer_names: List[Any]
    ) -> None:
        """
        Create multiplex network from one nx.Graph layer and layers names.

        Note that `network_layer` is replicated through all layers.

        :param network_layer: basic layer which is replicated through all ones
        :param layer_names: names for layers in multiplex network
        """
        # clear former layers of the object
        self.layers = {}

        for name in layer_names:
            self.layers.update({name: deepcopy(network_layer)})

        self._prepare_nodes_attribute()

    def _prepare_nodes_attribute(self) -> None:
        """Prepare network to the experiment."""
        for layer in self.layers.values():
            status_dict = {n: None for n in layer.nodes()}
            nx.set_node_attributes(layer, status_dict, "status")

    def describe(self, to_print: bool = True) -> Optional[str]:
        """
        Print out quickly parameters of the object.

        :param to_print: a flag, if true string is printed out to the console

        :return: produced string (if to_print is False)
        """
        assert len(self.layers) > 0, "Import network to the object first!"
        final_str = (
            "============================================\n"
            "network parameters\n"
            "--------------------------------------------\n"
        )

        final_str += "general parameters:\n"
        final_str += f"\tnumber of layers: {len(self.layers)}\n"
        mul_coeff = self._compute_multiplexing_coefficient()
        final_str += f"\tmultiplexing coefficient: {round(mul_coeff, 4)}\n"

        for name, graph in self.layers.items():
            final_str += f"\nlayer '{name}' parameters:\n"
            final_str += (
                f"\tgraph type - {type(graph)}\n\tnumber of nodes - "
                f"{len(graph.nodes())}\n\tnumber of edges - "
                f"{len(graph.edges())}\n"
            )
            avg_deg = np.average([_[1] for _ in graph.degree()])
            final_str += f"\taverage degree - {round(avg_deg, 4)}\n"
            if len(graph) > 0:
                final_str += (
                    f"\tclustering coefficient - "
                    f"{round(nx.average_clustering(graph), 4)}\n"
                )
            else:
                final_str += "\tclustering coefficient - nan\n"
        final_str += "============================================"

        if to_print:
            print(final_str)
            return None

        return final_str

    def _compute_multiplexing_coefficient(self) -> float:
        """
        Compute multiplexing coefficient.

        Multiplexing coefficient is defined as proportion of number of nodes
        common to all layers to number of all unique nodes in entire network

        :return: (float) multiplexing coefficient
        """
        assert len(self.layers) > 0, "Import network to the object first!"

        # create set of all nodes in the network
        all_nodes = set()
        for graph in self.layers.values():
            all_nodes.update([*graph.nodes()])

        # create set of all common nodes in the network over the layers
        common_nodes = set()
        for node in all_nodes:
            _ = True
            for layer in self.layers.values():
                if node not in layer:
                    _ = False
                    break
            if _ is True:
                common_nodes.add(node)

        if len(all_nodes) > 0:
            return len(common_nodes) / len(all_nodes)
        else:
            return 0

    def get_nodes_states(self) -> Dict[str, Any]:
        """
        Return summary of network's nodes states.

        :return: dictionary with items representing each of layers and with
            summary of node states in values
        """
        assert len(self.layers) > 0, "Import network to the object first!"
        statistics = {}
        for name, layer in self.layers.items():
            tab = []
            for node in layer.nodes():
                tab.append(layer.nodes[node]["status"])
            statistics[name] = tuple(Counter(tab).items())
        return statistics

    def get_node_state(self, node: Any) -> Tuple[str, ...]:
        """
        Return general state of node in network.

        In format acceptable by PropagationModel object, which means a tuple
        of sorted strings in form <layer name>.<state>

        :param node: a node which status has to be returned

        :return: tuple with states of node for each layer
        """
        assert len(self.layers) > 0, "Import network to the object first!"
        statistics = []
        for name, layer in self.layers.items():
            if node in layer:
                statistics.append(f"{name}.{layer.nodes[node]['status']}")
        statistics = sorted(statistics)
        return tuple(statistics)

    # TODO - probably deprecated
    def get_neighbours_attributes(
        self, node: Any, layer: str
    ) -> List[Dict[str, Any]]:
        """
        Return attributes in given layer of given node's neighbours.

        :param node: name of node
        :param layer: layer from which neighbours are being examined

        :return: list of neighbours' attributes
        """
        assert len(self.layers) > 0, "Import network to the object first!"
        attributes = []
        neighbours = [*self.layers[layer].neighbors(node)]
        for neighbour in neighbours:
            attributes.append(self.layers[layer].node[neighbour])  # TODO
        return attributes

    def get_layer_names(self) -> List[str]:
        """
        Get names of layers in the network.

        :return: list of layers' names
        """
        return [*self.layers.keys()]
