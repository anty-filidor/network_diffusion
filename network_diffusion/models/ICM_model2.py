

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
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
    STOP = 0
    NODE_TO_EXECUTE = Dict
    probability =0

    def __init__(
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        protocol: str,
        probability: float
    ) -> None:
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
        #descr += self._compartmental_graph.describe()
        descr += str(self._seed_selector)
        descr += f"{BOLD_UNDERLINE}\nauxiliary parameters\n{THIN_UNDERLINE}"
        descr += f"\n\tprotocole: {self.protocol.__name__}"
        descr += f"\n\tactive state abbreviation: {self.ACTIVE_NODE}"
        descr += f"\n\tinactive state abbreviation: {self.INACTIVE_NODE}"
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def _create_comparents(
            self,
            sending_budget: Tuple[NumericType, NumericType],
    ) -> CompartmentalGraph:

        compart_graph = CompartmentalGraph()
       # assert 0 <= probality <= 1, f"incorrect probality: {probality}!"

        compart_graph.add(process_name=self.PROCESS_NAME, states=[self.ACTIVE_NODE,self.INACTIVE_NODE])
        compart_graph.seeding_budget = {self.PROCESS_NAME: sending_budget}


        return  compart_graph
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
            net= net,
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
        probability =self.probability
        l_graph: nx.Graph = net.layers[layer_name]
        current_state = l_graph.nodes[agent.actor_id]["status"]
        if current_state == self.ACTIVE_NODE:
            return current_state


        for neighbour in l_graph.neighbors(agent.actor_id):
            if l_graph.nodes[neighbour]:
                r = random.random()
                if r < probability:
                    return self.ACTIVE_NODE
        return current_state



    def network_evaluation_step(
        self, net: MultilayerNetwork
    ) -> List[NUBff]:
        self.NODE_TO_EXECUTE = net.get_actors()
        activated_nodes: List[NUBff] = []
       # layer_inputs ={}
       # active_nodes = self.symulacja(net)

        for actor in net.get_actors():
            layer_inputs = {}

            if set(actor.states.values()) == set(self.ACTIVE_NODE):
                continue
            for layer_name in actor.layers:
                layer_inputs[layer_name] = self.agent_evaluation_step(actor,layer_name,net)
            if self.protocol(layer_inputs):
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






    def symulacja(self, net:MultilayerNetwork) -> List[NUBff]:
        probability = self.probability
        node_to_sprw = []
        activated_nodes: List[NUBff] = []
        node_to_execute = net.get_actors()

        for layer_name in net.layers:

            l_graph: nx.Graph = net.layers[layer_name]
            #node_to_execute = l_graph.nodes
            for node in node_to_execute:

                if l_graph.nodes[node.actor_id]["status"] == self.ACTIVE_NODE: #sprawdzenie czy wezeł aktywny
                    for neighbour in l_graph.neighbors(node.actor_id):
                        if l_graph.nodes[neighbour]["status"] == self.ACTIVE_NODE: # sprawdzenie czy sąsiad nie jeest czasem aktywny
                            continue
                        r = random.random()
                        if r < probability:  # warunek prawdopodobieństwa
                            # = self.ACTIVE_NODE # poprawić
                            activated_nodes.extend(
                            [NUBff( # uaktualnienie stanu wezła globalnie (wpływa na pętle?)
                                node_name=neighbour,
                                layer_name=layer_name,
                                new_state=self.ACTIVE_NODE
                            )])
                            node_to_sprw.append(neighbour) # dodanie do listy wykonującej
                   # l_graph.nodes[node.actor_id]["status"] = self.DONE # poprawić

            while len(node_to_sprw) != 0: # procedura dla sasiadów zapisanych w liście
                for node_n in node_to_sprw:
                    a = l_graph.nodes[node_n]["status"]
                    if l_graph.nodes[node_n]["status"] != self.ACTIVE_NODE:

                        node_to_sprw.remove(node_n)
                        continue
                    for neighbour_n in l_graph.neighbors(node_n):
                        if l_graph.nodes[neighbour_n]["status"] == self.ACTIVE_NODE: # sprawdzenie czy sąsiad nie jest już aktywny
                            continue
                        r = random.random()
                        if r < probability:  # warunek prawdopodobieństwa
                                l_graph.nodes[neighbour_n]["status"] = self.ACTIVE_NODE
                                activated_nodes.extend(
                                    [NUBff(  # uaktualnienie stanu wezła globalnie (wpływa na pętle?)
                                        node_name=neighbour_n,
                                        layer_name=layer_name,
                                        new_state=self.ACTIVE_NODE
                                    )])
                                node_to_sprw.append(neighbour_n)
                    node_to_sprw.remove(node_n)
            '''for node_udp in node_to_execute:
                if l_graph.nodes[node_udp.actor_id]["status"] == self.DONE:
                    l_graph.nodes[node_udp.actor_id]["status"] = self.ACTIVE_NODE # poprawić'''
        return  activated_nodes

    def get_allowed_states(self, net: MultilayerNetwork) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states in each layer of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {l_name: cmprt for l_name in net.layers}


