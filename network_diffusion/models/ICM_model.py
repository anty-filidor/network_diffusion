
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
    DONE = "-1"
    NODE_TO_EXECUTE = Dict
    probability = 0

    def __init__(
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        protocol: str,
        probability: float
    ) -> None:
        self.probability = probability

        assert 0 <= probability <= 1, f"incorrect probability: {probability}!"

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
        # function only check state of actor

        l_graph: nx.Graph = net.layers[layer_name]
        current_state = l_graph.nodes[agent.actor_id]["status"]
        return current_state

    def network_evaluation_step(
         self,
         net: MultilayerNetwork
    ) -> List[NUBff]:

        self.NODE_TO_EXECUTE = net.get_actors()
        activated_nodes: List[NUBff] = []
        active_nodes = self.simulation(net)

        for actor in net.get_actors():
            layer_inputs = {}

            for layer_name in actor.layers:  # checking each layer in site
                # saving state of each node's layer to layer_inputs
                layer_inputs[layer_name] = self.agent_evaluation_step(actor, layer_name, net)
                if self.protocol(layer_inputs):  # checking conditions for selected protocol
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

    def simulation(self, net: MultilayerNetwork) -> List[NUBff]:
        """

        Return list of activated nodes in this simulation

        """

        probability = self.probability
        node_to_check = []  # list of nodes to check in next turn
        activated_nodes: List[NUBff] = []
        node_to_execute = net.get_actors()  # loading nodes to simulation

        for layer_name in net.layers:

            l_graph: nx.Graph = net.layers[layer_name]  # choosing a layer of site

            for node in node_to_execute:  # choosing of node
                # checking a state of node, only active node can do next steps
                if l_graph.nodes[node.actor_id]["status"] == self.ACTIVE_NODE:
                    for neighbour in l_graph.neighbors(node.actor_id):
                        # checking a state of neighbour of node
                        if l_graph.nodes[neighbour]["status"] == self.ACTIVE_NODE:
                            continue  # if neighbour is active leave it
                        r = random.random()  # randomization
                        if r < probability:  # condition of probability
                            # status is changed on active
                            l_graph.nodes[neighbour]["status"] = self.ACTIVE_NODE
                            # saving to the activated nodes
                            activated_nodes.extend(
                             [NUBff(
                                node_name=neighbour,
                                layer_name=layer_name,
                                new_state=self.ACTIVE_NODE
                             )])
                            # adding neighbour to the next list
                            node_to_check.append(neighbour)
                            # checking a node's state on DONE if finish its proces
                    l_graph.nodes[node.actor_id]["status"] = self.DONE

            # the nest turn for activated neighbours

            while len(node_to_check) != 0:  # list can't be null
                for node_n in node_to_check:

                    # node's state must be active
                    if l_graph.nodes[node_n]["status"] != self.ACTIVE_NODE:
                        node_to_check.remove(node_n)   # removing the node after verification
                        continue

                    for neighbour_n in l_graph.neighbors(node_n):  # checking the neighborhood
                        # checking if neighbour is activated
                        if l_graph.nodes[neighbour_n]["status"] == self.ACTIVE_NODE:
                            continue

                        r = random.random()  # randomization
                        if r < probability:  # condition of probability
                            # changing the node's state on active
                            l_graph.nodes[neighbour_n]["status"] = self.ACTIVE_NODE
                            activated_nodes.extend(
                                    [NUBff(
                                        node_name=neighbour_n,
                                        layer_name=layer_name,
                                        new_state=self.ACTIVE_NODE
                                    )])
                            node_to_check.append(neighbour_n)  # adding neighbour on the end of list
                    node_to_check.remove(node_n)  # removing node after proces
            for node_udp in node_to_execute:  # changing node statuses from DONE to ACTIVE
                if l_graph.nodes[node_udp.actor_id]["status"] == self.DONE:
                    l_graph.nodes[node_udp.actor_id]["status"] = self.ACTIVE_NODE
        return activated_nodes

    def get_allowed_states(self, net: MultilayerNetwork) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {l_name: cmprt for l_name in net.layers}
