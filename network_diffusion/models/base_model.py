# Copyright (c) 2023 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Definition of the base propagation model used in the library."""

from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Tuple

from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.seeding.base_selector import BaseSeedSelector


class BaseModel(ABC):
    """Base abstract propagation model."""

    @property
    @abstractmethod
    def _compartmental_graph(self) -> CompartmentalGraph:
        """Compartmental model that defines allowed transitions and states."""

    @property
    @abstractmethod
    def _seed_selector(self) -> BaseSeedSelector:
        """A method of selecting seed agents."""

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of the object."""

    @abstractmethod
    def determine_initial_states(
        self, net: MultilayerNetwork
    ) -> List[NetworkUpdateBuffer]:
        """
        Determine initial states in the network according to seed selector.

        :param net: network to initialise seeds for
        :return: list of nodes with their states
        """

    @abstractmethod
    def agent_evaluation_step(
        self,
        agent: Any,
        layer_name: str,
        net: MultilayerNetwork,
    ) -> str:
        """
        Try to change state of given node of the network according to model.

        :param agent: id of the node or the actor to evaluate
        :param layer_name: a layer where the node exists
        :param net: a network where the node exists

        :return: state of the model after evaluation
        """

    @abstractmethod
    def network_evaluation_step(
        self, net: MultilayerNetwork
    ) -> List[NetworkUpdateBuffer]:
        """
        Evaluate the network at one time stamp according to the model.

        :param network: a network to evaluate
        :return: list of nodes that changed state after the evaluation
        """

    @staticmethod
    def update_network(
        net: MultilayerNetwork,
        activated_nodes: List[NetworkUpdateBuffer],
    ) -> List[Dict[str, str]]:
        """
        Update the network global state by list of already activated nodes.

        :param net: network to update
        :param activated_nodes: already activated nodes
        """
        out_json = []
        for active_node in activated_nodes:
            net.layers[active_node.layer_name].nodes[active_node.node_name][
                "status"
            ] = active_node.new_state
            out_json.append(active_node.to_json())
        return out_json

    @abstractmethod
    def get_allowed_states(
        self, net: MultilayerNetwork
    ) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """

    @staticmethod
    def get_states_num(
        net: MultilayerNetwork,
    ) -> Dict[str, Tuple[Tuple[Any, int], ...]]:
        """
        Return states in the network with number of agents that adopted them.

        It is the most basic function which assumes that field "status" in the
        network is self explaining and there is no need to decode it (e.g.
        to separate hidden state from public one).

        :return: dictionary with items representing each of layers and with
            summary of nodes states in values
        """
        statistics = {}
        for name, layer in net.layers.items():
            tab = []
            for node in layer.nodes():
                tab.append(layer.nodes[node]["status"])
            statistics[name] = tuple(Counter(tab).items())
        return statistics
