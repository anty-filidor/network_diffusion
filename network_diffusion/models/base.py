"""Definition of the base propagation model used in the library."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.multilayer_network import MultilayerNetwork


@dataclass
class NetworkUpdateBuffer:
    """Auxiliary class to keep info about nodes that needs to be udated."""

    node_name: str
    layer_name: str
    new_state: str

    def __str__(self) -> str:
        return f"{self.layer_name}:{self.node_name}:{self.new_state}"


class BaseModel(ABC):
    """Base abstract propagation model."""

    def __init__(self, compartmental_graph: CompartmentalGraph) -> None:
        """
        Create the object.

        :param compartmental_graph: a compartmental model that defines
            allowed transitions and states
        """
        self._compartmental_graph = compartmental_graph

    @property
    def compartments(self) -> CompartmentalGraph:
        """Return defined compartments and allowed transitions."""
        return self._compartmental_graph

    @abstractmethod
    def node_evaluation_step(
        self, node_id: int, layer_name: str, net: MultilayerNetwork
    ) -> str:
        """
        Try to change state of given node of the network according to model.

        :param node_id: id of the node to evaluate
        :param layer_name: a layer where the node exists
        :param network: a network where the node exists

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
