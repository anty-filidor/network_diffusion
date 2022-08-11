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

"""Functions for the phenomena spreading definition."""

# pylint: disable=W0141

from copy import deepcopy
from random import shuffle
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from network_diffusion.experiment_logger import ExperimentLogger
from network_diffusion.multilayer_network import MultilayerNetwork
from network_diffusion.propagation_model import PropagationModel


class MultiSpreading:
    """Perform experiment defined by PropagationModel on MultiLayerNetwork."""

    def __init__(
        self, model: PropagationModel, network: MultilayerNetwork
    ) -> None:
        """
        Construct an object..

        :param model: model of propagation which determines how experiment
            looks like
        :param network: a network which is being examined during experiment
        """
        assert network.layers.keys() == model.get_model_hyperparams().keys(), (
            "Layer names in network should be the same as layer names in "
            "propagation model"
        )
        self._model = model
        self._network = deepcopy(network)

    def set_initial_states(
        self,
        states_seeds: Dict[str, Tuple[int, ...]],
        track_changes: bool = False,
    ) -> None:
        """
        Prepare network to be used during simulation.

        It changes status of certain nodes as passed in states_seeds. After
        execution of this method experiment is reeady to be done.

        :param states_seeds: dictionary with numbers of nodes to be initialised
            in each layer of network. For example following argument:
            {
                illness': (75, 2, 0),
                'awareness': (60, 17),
                'vaccination': (70, 7)
            }
            is correct with this model of propagation:
            [
                illness : ('S', 'I', 'R'),
                awareness : ('UA', 'A'),
                vaccination : ('UV', 'V')
            ].
            Note, that values in dictionary says how many nodes should have
             state <x> at the beginning of the experiment.
        :param track_changes: flag, if true changes are being printed out
        """
        # pylint: disable=R0914

        model_hyperparams = self._model.get_model_hyperparams()

        # check if argument has good shape
        ass_states_arg = [len(s) for s in states_seeds.values()]
        ass_states_net = [len(s) for s in model_hyperparams.values()]
        assert ass_states_net == ass_states_arg, (
            f"Shape of argument {ass_states_arg} should be the same as shape "
            f"of states in propagation model {ass_states_net}"
        )

        # check if given argument has good keys
        assert states_seeds.keys() == model_hyperparams.keys(), (
            "Layer names in argument should be the same as layer names in "
            "propagation model"
        )

        # check if given argument has good values
        ass_states_sum = [
            sum(s[1]) == len(self._network.layers[s[0]].nodes())
            for s in states_seeds.items()
        ]
        for ssum in ass_states_sum:
            assert ssum, (
                "Sum of size of states for each layer should be equal to "
                "number of its nodes!"
            )

        # set initial states in each layer of network
        for name, layer in self._network.layers.items():
            states = model_hyperparams[name]
            vals = states_seeds[name]
            nodes_number = len(layer.nodes())
            if track_changes:
                print(name, states, vals, nodes_number)

            # shuffle nodes
            nodes = [*layer.nodes()]
            shuffle(nodes)

            # set ranges
            _rngs = [sum(vals[:x]) for x in range(len(vals))] + [nodes_number]
            ranges: List[Tuple[int, int]] = list(zip(_rngs[:-1], _rngs[1:]))
            if track_changes:
                print(ranges)

            # append states to nodes
            for i, _ in enumerate(ranges):
                pair = ranges[i]
                state = states[i]
                if track_changes:
                    print(pair, state)
                for index in range(pair[0], pair[1]):
                    if track_changes:
                        print(index, nodes[index], layer.node[nodes[index]])
                    layer.nodes[nodes[index]]["status"] = state
                    if track_changes:
                        print(index, nodes[index], layer.node[nodes[index]])

    def perform_propagation(self, n_epochs: int) -> ExperimentLogger:
        """
        Perform experiment on given network and given model.

        It saves logs in ExperimentLogger object which can be used for further
        analysis.

        :param n_epochs: number of epochs to do experiment
        :return: logs of experiment stored in special object
        """
        # pylint: disable=W0212, R1702
        logger = ExperimentLogger(
            self._model._get_description_str(),
            self._network._get_description_str(),
        )

        # add logs from initial state
        logger._add_log(self._network.get_nodes_states())

        # iterate through epochs
        progress_bar = tqdm(range(n_epochs))
        for epoch in progress_bar:
            progress_bar.set_description_str(f"Processing epoch {epoch}")

            # in each epoch iterate through layers
            for layer_name, layer_graph in self._network.layers.items():

                # in each layer iterate through nodes
                for n in layer_graph.nodes():

                    # read state of node
                    state = self._network.get_node_state(n)

                    # import possible transitions for state of the node
                    possible_transitions = (
                        self._model.get_possible_transitions(state, layer_name)
                    )

                    # if there is no possible transition don't do anything,
                    # otherwise:
                    if len(possible_transitions) > 0:

                        # iterate through neighbours of current node
                        for nbr in nx.neighbors(layer_graph, n):
                            status = layer_graph.nodes[nbr]["status"]

                            # if state of neighbour node is in possible
                            # transitions and tossed 1 due to it's weight
                            if (
                                status in possible_transitions
                                and np.random.choice(
                                    [0, 1],
                                    p=[
                                        1 - possible_transitions[status],
                                        possible_transitions[status],
                                    ],
                                )
                                == 1
                            ):
                                # change state of current node and breag
                                # iterating through neighbours
                                layer_graph.nodes[n][
                                    "status"
                                ] = layer_graph.nodes[nbr]["status"]
                                break

            # add logs from current epoch
            logger._add_log(self._network.get_nodes_states())

        # convert logs to dataframe
        logger._convert_logs(self._model.get_model_hyperparams())

        return logger
