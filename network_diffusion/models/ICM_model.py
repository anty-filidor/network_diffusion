
from typing import Dict, List, Tuple
import random
import numpy as np
import networkx as nx
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer as NUBff
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType


class ICModel(BaseModel):

    INACTIVE_NODE = "0"
    ACTIVE_NODE = "1"
    PROCESS_NAME = "ICM"
    ACTIVATED_NODE = "-1"
    probability = 0

    def __init__(
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        protocol: str,
        probability: float
    ) -> None:
        assert 0 <= probability <= 1, f"incorrect probability: {probability}!"
        self.probability = probability
        compart_graph = self._create_comparents(seeding_budget)
        super().__init__(compart_graph, seed_selector)
        if protocol == "AND":
            self.protocol = self._protocol_and

        elif protocol == "OR":
            self.protocol = self._protocol_or

        else:
            raise ValueError("Only AND & OR value is allowed!")

    def __str__(self) -> str:
        """Return string representation of the object."""
        descr = f"{BOLD_UNDERLINE}\nMultilayer Independent Cascade Model"
        descr += f"\n{THIN_UNDERLINE}\n"
        descr += str(self._seed_selector)
        descr += f"{BOLD_UNDERLINE}\nauxiliary parameters\n{THIN_UNDERLINE}"
        descr += f"\n\tprotocols: {self.protocol.__name__}"
        descr += f"\n\tactive state abbreviation: {self.ACTIVE_NODE}"
        descr += f"\n\tinactive state abbreviation: {self.INACTIVE_NODE}"
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def _create_comparents(
            self,
            sending_budget: Tuple[NumericType, NumericType],
    ) -> CompartmentalGraph:

        compart_graph = CompartmentalGraph()
        compart_graph.add(process_name=self.PROCESS_NAME, states=[self.ACTIVE_NODE, self.INACTIVE_NODE])
        compart_graph.seeding_budget = {self.PROCESS_NAME: sending_budget}

        return compart_graph

    @staticmethod
    def _protocol_or(inputs: Dict[str, str]) -> bool:
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.any())

    @staticmethod
    def _protocol_and(inputs: Dict[str, str]) -> bool:
        inputs_bool = np.array([bool(int(input)) for input in inputs.values()])
        return bool(inputs_bool.all())

    def set_initial_states(
        self, net: MultilayerNetwork
    ) -> List[Dict[str, str]]:

        budget = self._compartmental_graph.get_seeding_budget_for_network(
            net=net,
            actorwise=True
        )
        seed_nodes: List[NUBff] = []
        for idx, actor in enumerate(self._seed_selector.actorwise(net=net)):
            if idx < budget[self.PROCESS_NAME][self.ACTIVE_NODE]:
                a_state = self.ACTIVE_NODE
            else:
                a_state = self.INACTIVE_NODE

            for l_name in actor.layers:
                seed_nodes.append(
                    NUBff(
                        node_name=actor.actor_id,
                        layer_name=l_name,
                        new_state=a_state
                    )
                )
        out_json = self.update_network(net=net, activated_nodes=seed_nodes)
        return out_json

    def agent_evaluation_step(
        self,
        agent: MLNetworkActor,
        layer_name: str,
        net: MultilayerNetwork

    ) -> str:
        """

        param_agent is node to check
        param_layer_name is node's layer
        param_net is choosing network

        """

        l_graph: nx.Graph = net.layers[layer_name]
        current_state = l_graph.nodes[agent.actor_id]["status"]
        if current_state == self.ACTIVE_NODE:
            return self.ACTIVATED_NODE
        # checking neighborhood of actor
        for neighbour in l_graph.neighbors(agent.actor_id):
            # if neighbour is active, it can send signal to activation of actor
            if l_graph.nodes[neighbour]["status"] == self.ACTIVE_NODE:
                # random is homogeneous distribution
                r = random.random()
                # actor is activated if a probability allow its
                if r < self.probability:
                    return self.ACTIVE_NODE
        return current_state

    def network_evaluation_step(
        self, net: MultilayerNetwork
    ) -> List[NUBff]:

        """
        param_net: load network
        return a list of activated_nodes if the protocol allow

        """

        activated_nodes: List[NUBff] = []

        for actor in net.get_actors():  # checking each node
            layer_inputs = {}

            if set(actor.states.values()) == set(self.ACTIVATED_NODE):
                continue
            for layer_name in actor.layers:
                # downloading the status of actor
                layer_inputs[layer_name] = self.agent_evaluation_step(actor, layer_name, net)
            if self.protocol(layer_inputs):  # checking the protocol's conditions
                # adding to list
                activated_nodes.extend(
                    [
                        NUBff(
                            node_name=actor.actor_id,
                            layer_name=layer_name,
                            new_state=self.ACTIVE_NODE,
                        )
                        for layer_name in layer_inputs
                    ]
                )
        return activated_nodes

    def get_allowed_states(self, net: MultilayerNetwork) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {l_name: cmprt for l_name in net.layers}
