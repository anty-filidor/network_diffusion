"""Definition of the base propagation model used in the library."""

from abc import ABC, abstractmethod
from typing import Any, List

from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.multilayer_network import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector


class BaseModel(ABC):
    """Base abstract propagation model."""

    def __init__(
        self,
        compartmental_graph: CompartmentalGraph,
        seed_selector: BaseSeedSelector,
    ) -> None:
        """
        Create the object.

        :param compartmental_graph: a compartmental model that defines allowed
            transitions and states
        :param seed_selector: definition of the influence seeds choice
        """
        self._compartmental_graph = compartmental_graph
        self._seed_selector = seed_selector
        self._seeds: List[NetworkUpdateBuffer] = []

    @property
    def compartments(self) -> CompartmentalGraph:
        """Return defined compartments and allowed transitions."""
        return self._compartmental_graph

    def set_initial_states(self, net: MultilayerNetwork) -> MultilayerNetwork:
        """
        Set up network to reflect given initial states in the model.

        :param net: a network where the node exists
        """
        if len(self._seeds) == 0:
            self._seeds = self._seed_selector(self._compartmental_graph, net)
        raise NotImplementedError

    @abstractmethod
    def node_evaluation_step(
        self, node_id: int, layer_name: str, net: MultilayerNetwork
    ) -> str:
        """
        Try to change state of given node of the network according to model.

        :param node_id: id of the node to evaluate
        :param layer_name: a layer where the node exists
        :param net: a network where the node exists

        :return: state of the model after evaluation
        """
        ...

    @abstractmethod
    def network_evaluation_step(
        self, net: MultilayerNetwork
    ) -> List[NetworkUpdateBuffer]:
        """
        Evaluate the network at one time stamp according to the model.

        :param network: a network to evaluate
        :return: list of nodes that changed state after the evaluation
        """
        ...

    @staticmethod
    def update_network(
        net: MultilayerNetwork,
        activated_nodes: List[NetworkUpdateBuffer],
    ) -> None:
        """
        Update the network global state by list of already activated nodes.

        :param network: network to update
        :param activated_nodes: already activated nodes
        """
        for activ_node in activated_nodes:
            net.layers[activ_node.layer_name].nodes[activ_node.node_name][
                "status"
            ] = activ_node.new_state
