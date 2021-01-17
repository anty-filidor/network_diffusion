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
from copy import deepcopy
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from network_diffusion.multilayer_network import MultilayerNetwork
from network_diffusion.propagation_model import PropagationModel
from network_diffusion.utils import create_directory


class ExperimentLogger:
    """Stores and processes logs acquired during performing MultiSpreading."""

    def __init__(
        self, model_description: str, network_description: str
    ) -> None:
        """
        Default constructor.

        :param model_description: description of the model (i.e.
            PropagationModel.describe()) which is used for saving in logs
        :param network_description: description of the network (i.e.
            MultiplexNetwork.describe()) which is used for saving in logs
        """
        self._model_description = model_description
        self._network_description = network_description
        self._raw_stats: List[Dict[str, Any]] = []
        self._stats: Optional[Dict[str, Any]] = None

    def _add_log(self, log: Dict[str, Any]) -> None:
        """
         Adds raw log from single epoch to the object.

        :param log: raw log (i.e. a single call of
            MultiplexNetwork.get_nodes_states())
        """
        self._raw_stats.append(log)

    def _convert_logs(
        self, model_parameters: Dict[str, Tuple[Dict[str, Any]]]
    ) -> None:
        """
        Converts raw logs into pandas dataframe.

        Used after finishing aggregation of logs. It fulfills self._stats.

        :param model_parameters: parameters of the propagation model to store
        """
        # initialise container for splatted data
        self._stats = {
            k: pd.DataFrame(columns=model_parameters[k])
            for k in model_parameters.keys()
        }

        # fill containers
        for epoch in self._raw_stats:
            for layer, vals in epoch.items():
                self._stats[layer] = self._stats[layer].append(
                    dict(vals), ignore_index=True
                )

        # change NaN values to 0 and all values to integers
        for layer, vals in self._stats.items():
            self._stats[layer] = vals.fillna(0).astype(int)

    def __str__(self) -> str:
        """Allows to print out object."""
        return str(self._stats)

    def plot(self, to_file: bool = False, path: Optional[str] = None) -> None:
        """
        Plots out visualisation of performed experiment.

        :param to_file: flag, if true save figure to file, otherwise it
            is plotted on screen
        :param path: path to save figure
        """
        fig = plt.figure()

        for i, layer in enumerate(self._stats, 1):
            ax = fig.add_subplot(len(self._stats), 1, i)
            self._stats[layer].plot(ax=ax, legend=True)
            ax.set_title(layer)
            ax.legend(loc="upper right")
            ax.set_ylabel("Nodes")
            if i == 1:
                nn = self._stats[layer].iloc[0].sum()
            ax.set_yticks(np.arange(0, nn + 1, 20))
            if i == len(self._stats):
                ax.set_xlabel("Epoch")
            ax.grid()

        plt.tight_layout(0.3)
        if to_file:
            plt.savefig(f"{path}/visualisation.png", dpi=200)
        else:
            plt.show()

    def report(
        self,
        visualisation: bool = False,
        to_file: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """
        Creates report of experiment.

        It consists of report of the network, report of the model, record of
        propagation progress and optionally visualisation of the progress.

        :param visualisation: (bool) a flag, if true visualisation is being
            plotted
        :param to_file: a flag, if true report is saved in files,
            otherwise it is printed out on the screen
        :param path: (str) path to folder where report will be saved
        """
        if to_file:
            # create directory from given path
            create_directory(path)
            # save progress in propagation of each layer to csv file
            for stat in self._stats:
                self._stats[stat].to_csv(
                    path + "/" + stat + "_propagation_report.csv",
                    index_label="epoch",
                )
            # save description of model to txt file
            with open(path + "/model_report.txt", "w") as file:
                file.write(self._model_description)
            # save description of network to txt file
            with open(path + "/network_report.txt", "w") as file:
                file.write(self._network_description)
            # save figure
            if visualisation:
                self.plot(to_file=True, path=path)
        else:
            print(self._network_description)
            print(self._model_description)
            print(
                "============================================\n"
                "propagation report\n"
                "--------------------------------------------"
            )
            for stat in self._stats:
                print(stat, "\n", self._stats[stat], "\n")
            print("============================================")
            if visualisation:
                self.plot()


class MultiSpreading:
    """Performs experiment defined by PropagationModel on MultiLayerNetwork."""

    def __init__(
        self, model: PropagationModel, network: MultilayerNetwork
    ) -> None:
        """
        Default constructor.

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
        Prepares network to be used during simulation.

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
            N = len(layer.nodes())
            if track_changes:
                print(name, states, vals, N)

            # shuffle nodes
            nodes = [*layer.nodes()]
            shuffle(nodes)

            # set ranges
            ranges = [sum(vals[:x]) for x in range(len(vals))]
            ranges.append(N)
            ranges = [(i, j) for i, j in zip(ranges[:-1], ranges[1:])]
            if track_changes:
                print(ranges)

            # append states to nodes
            for i in range(len(ranges)):
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
        Performs experiment on given network and given model.

        It saves logs in ExperimentLogger object which can be used for further
        analysis.

        :param n_epochs: number of epochs to do experiment
        :return: logs of experiment stored in special object
        """
        # initialise container for logs
        logger = ExperimentLogger(
            self._model.describe(to_print=False),
            self._network.describe(to_print=False),
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
                    possible_transitions = self._model.get_possible_transitions(
                        state, layer_name
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
