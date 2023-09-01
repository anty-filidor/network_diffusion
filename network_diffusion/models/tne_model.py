# Copyright 2023 by Michał Czuba, Piotr Bródka. All Rights Reserved.
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

"""Definition of the temporal network epistemology model."""

from typing import Dict, List, Tuple, Any

import networkx as nx
import numpy as np

from network_diffusion.tpn.tpnetwork import TemporalNetwork
from network_diffusion.models.base_tp_model import BaseTPModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE, NumericType


class TemporalNetworkEpistemologyModel(BaseTPModel):
    """
    This model implements generalized version of Temporal Network Epistemology Model.
    """

    PROCESS_NAME = "TNEM"
    A_STATE = "A"
    B_STATE = "B"

    def __init__(
        self,
        seeding_budget: Tuple[NumericType, NumericType],
        seed_selector: BaseSeedSelector,
        trials_nr: int,
        epsilon: float
    ) -> None:
        """
        Create the object.

        :param seeding_budget: a proportion of INACTIVE and ACTIVE nodes in each layer
        :param seed_selector: class that selects initial seeds for simulation
        :param trials_nr: number of trials an agent performs when experimenting with environment by drawing a sample
            from the binomial distribution
        :param epsilon: the difference between expected value of B action, and A action equal to 0.5
        """
        compart_graph = self._create_compartments(seeding_budget)
        super().__init__(compart_graph, seed_selector)
        self.trials_nr = trials_nr
        self.epsilon = epsilon

    def __str__(self) -> str:
        """Return string representation of the object."""
        descr = f"{BOLD_UNDERLINE}\nTemporal Network Epistemology Model"
        descr += f"\n{THIN_UNDERLINE}\n"
        descr += self._compartmental_graph.describe()
        descr += f"\n{self._seed_selector}"
        descr += f"{BOLD_UNDERLINE}\nauxiliary parameters\n{THIN_UNDERLINE}"
        descr += f"\n{BOLD_UNDERLINE}"
        return descr

    def _create_compartments(
        self,
        seeding_budget: Tuple[NumericType, NumericType]
    ) -> CompartmentalGraph:
        """
        Create compartmental graph for the model.

        Create one process with two states: A, B.
        """
        compart_graph = CompartmentalGraph()

        # Add allowed states and seeding budget
        compart_graph.add(
            process_name=self.PROCESS_NAME,
            states=[self.A_STATE, self.B_STATE],
        )
        compart_graph.seeding_budget = {self.PROCESS_NAME: seeding_budget}

        # Add allowed transition
        compart_graph.compile()
        return compart_graph

    def set_initial_states(self, net: TemporalNetwork) -> List[Dict[str, str]]:
        """
        Set initial states in the network according to seed selection method.

        :param net: network to initialise seeds for

        :return: a list of state of the network after initialisation
        """
        budget = self._compartmental_graph.get_seeding_budget_for_network(net=net, actorwise=True)
        seed_nodes: List[NetworkUpdateBuffer] = []

        for idx, actor in enumerate(self._seed_selector.actorwise(net=net)):
            node_id, node = actor
            # select initial state for given actor according to budget
            if idx < budget[self.PROCESS_NAME][self.B_STATE]:
                state = self.B_STATE
                belief = np.random.uniform(0.5, 1)
                evidence = np.random.binomial(self.trials_nr, 0.5 + self.epsilon)
            else:
                state = self.A_STATE
                belief = np.random.uniform(0, 0.5)
                evidence = 0

            encoded_state = self.encode_actor_status(state, belief, evidence)

            # generate update buffer for the actor
            seed_nodes.append(
                NetworkUpdateBuffer(
                    node_name=node_id,
                    layer_name='TPN',
                    new_state=encoded_state
                )
            )

        # set initial states and return json to save in logs
        out_json = self.update_network(net=net, agents=seed_nodes, snapshot_id=0)
        return out_json

    @staticmethod
    def encode_actor_status(state: str, belief: float, evidence: int) -> str:
        """
        Encodes agent features to str form.

        :param state: state of an actor
        :param belief: level of agent's belief
        :param evidence: nr of successes drawn from binomial distribution in an experiment

        :return: a string representation of agent status
        """
        encoded_status = '_'.join([state, str(belief), str(evidence)])
        return encoded_status

    @staticmethod
    def decode_actor_status(encoded_status: str) -> Tuple[str, float, int]:
        """
        Decodes agent features from str form.

        :param encoded_status: a string representation of agent status

        :return: a tuple with agent state, belief level and evidence
        """
        # TODO: consider operating on dictionary instead of tuple
        state: str = encoded_status.split('_')[0]
        belief: float = float(encoded_status.split('_')[1])
        evidence: int = int(encoded_status.split('_')[2])
        return state, belief, evidence

    def agent_evaluation_step(self, agent: Any, net: TemporalNetwork, snapshot_id: int) -> str:
        """
        Try to change state of given actor of the network according to model.

        :param agent: actor to evaluate in given layer
        :param net: a network where the actor exists
        :param snapshot_id: currently processed snapshot

        :return: state of the actor to be set in the next snapshot
        """
        # TODO: consider unification of names - agents or actors?
        snapshot: nx.Graph = net.snaps[snapshot_id]
        node_id, node = agent

        # Gather evidence on action B performance
        state, belief, evidence = self.decode_actor_status(node["status"])
        neighbours_states = [self.decode_actor_status(snapshot.nodes[n]["status"]) for n in snapshot.neighbors(node_id)]
        b_neighbours_states = [state for state in neighbours_states if state[0] == self.B_STATE]
        n = len(b_neighbours_states) * self.trials_nr
        k = np.sum([states[2] for states in b_neighbours_states])
        if state == self.B_STATE:
            n += self.trials_nr
            k += evidence

        # Run credence update
        new_belief = 1 / (1 + (1 - belief) * (((0.5 - self.epsilon) / (0.5 + self.epsilon)) ** (2 * k - n)) / belief)

        if new_belief > 0.5:
            new_state = self.B_STATE
            new_evidence = np.random.binomial(self.trials_nr, 0.5 + self.epsilon)
        else:
            new_state = self.A_STATE
            new_evidence = 0

        encoded_state = self.encode_actor_status(new_state, new_belief, new_evidence)
        return encoded_state

    def network_evaluation_step(self, net: TemporalNetwork, snapshot_id: int) -> List[NetworkUpdateBuffer]:
        """
        Evaluate the network at one time stamp with MLTModel.

        :param net: a network to evaluate
        :param snapshot_id: currently processed snapshot

        :return: list of nodes that changed state after the evaluation
        """
        nodes_to_update: List[NetworkUpdateBuffer] = []

        for node_id, node in net.get_actors_from_snap(snapshot_id):
            actor = node_id, node
            new_state_encoded = self.agent_evaluation_step(actor, net, snapshot_id)
            nodes_to_update.append(
                NetworkUpdateBuffer(
                    node_name=node_id,
                    layer_name='TPN',
                    new_state=new_state_encoded
                )
            )
        return nodes_to_update

    def get_allowed_states(self, net: TemporalNetwork) -> Dict[str, Tuple[str, ...]]:
        """
        Return dict with allowed states of net if applied model.

        :param net: a network to determine allowed nodes' states for
        """
        cmprt = self._compartmental_graph.get_compartments()[self.PROCESS_NAME]
        return {'TPN': cmprt}
