from random import shuffle
from tqdm import tqdm
from copy import deepcopy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MultiSpreading:
    def __init__(self, model, network):
        assert network.layers.keys() == model.get_model_hyperparams().keys(), \
            'Layer names in network should be the same as layer names in propagation model'
        self._model = model
        self._network = deepcopy(network)

    def set_initial_states(self, states_seeds, track_changes=False):
        """
        This method prepares network to be used during simulation. It changes status of certain nodes following given
        seeds in function argument
        :param states_seeds: dictionary with numbers of nodes to be initialise in each layer of network.
                             For example following argument: {'illness': (75, 2, 0), 'awareness': (60, 17),
                             'vaccination': (70, 7)} is correct with this model of propagation:
                             [illness : ('S', 'I', 'R'), awareness : ('UA', 'A'), vaccination : ('UV', 'V')]. Note, that
                             values in dictionary says how many nodes should have state <x> at the beginning of the
                             experiment.
        :param track_changes: flag, if true changes are being printed out
        """

        model_hyperparams = self._model.get_model_hyperparams()

        # check if argument has good shape
        ass_states_arg = [len(s) for s in states_seeds.values()]
        ass_states_net = [len(s) for s in model_hyperparams.values()]
        assert ass_states_net == ass_states_arg, \
            'Shape of argument {} should be the same as shape of states in propagation model {}'\
            .format(ass_states_arg, ass_states_net)

        # check if given argument has good keys
        assert states_seeds.keys() == model_hyperparams.keys(), \
            'Layer names in argument should be the same as layer names in propagation model'

        # check if given argument has good values
        ass_states_sum = [sum(s[1]) == len(self._network.layers[s[0]].nodes()) for s in states_seeds.items()]
        for ssum in ass_states_sum:
            assert ssum, 'Sum of size of states for each layer should be equal to number of its nodes!'

        # set initial states in each layer of network
        for name, layer in self._network.layers.items():
            states = model_hyperparams[name]
            vals = states_seeds[name]
            N = len(layer.nodes())
            if track_changes: print(name, states, vals, N)

            # shuffle nodes
            nodes = [*layer.nodes()]
            shuffle(nodes)

            # set ranges
            ranges = [sum(vals[:x]) for x in range(len(vals))]
            ranges.append(N)
            ranges = [(i, j) for i, j in zip(ranges[:-1], ranges[1:])]
            if track_changes: print(ranges)

            # append states to nodes
            for i in range(len(ranges)):
                pair = ranges[i]
                state = states[i]
                if track_changes: print(pair, state)
                for index in range(pair[0], pair[1]):
                    if track_changes: print(index, nodes[index], layer.node[nodes[index]])
                    layer.node[nodes[index]]['status'] = state
                    if track_changes: print(index, nodes[index], layer.node[nodes[index]])

    def perform_propagation(self, n_epochs):
        """
        This method performs experiment on given network and given model. It saves logs in ExperimentLogger object
        :param n_epochs: number of epochs to do experiment
        :return: (ExperimentLogger) logs of experiment stored in special object
        """

        # initialise container for logs
        logger = ExperimentLogger()

        # add logs from initial state
        logger.add_log(self._network.get_nodes_states())

        # iterate through epochs
        progress_bar = tqdm(range(n_epochs))
        for epoch in progress_bar:
            progress_bar.set_description_str('Processing epoch {}'.format(epoch))

            # in each epoch iterate through layers
            for layer_name, layer_graph in self._network.layers.items():

                # in each layer iterate through nodes
                for n in layer_graph.nodes():

                    # read state of node
                    state = self._network.get_node_state(n)

                    # import possible transitions for state of the node
                    possible_transitions = self._model.get_possible_transitions(state, layer_name)

                    # if there is no possible transition don't do anything, otherwise:
                    if len(possible_transitions) > 0:

                        # iterate through neighbours of current node
                        for nbr in nx.neighbors(layer_graph, n):
                            status = layer_graph.node[nbr]['status']

                            # if state of neighbour node is in possible transitions and tossed 1 due to it's weight
                            if status in possible_transitions and \
                                    np.random.choice([0, 1], p=[1 - possible_transitions[status],
                                                                possible_transitions[status]]) == 1:
                                # change state of current node and breag iterating through neighbours
                                layer_graph.node[n]['status'] = layer_graph.node[nbr]['status']
                                break

            # add logs from current epoch
            logger.add_log(self._network.get_nodes_states())

        # add model description
        logger.add_model_description(self._model.get_model_hyperparams())

        return logger


class ExperimentLogger:
    def __init__(self):
        self._raw_stats = []
        self._model_description = None
        self._stats = None

    def add_log(self, log):
        self._raw_stats.append(log)

    def add_model_description(self, model_description):
        self._model_description = model_description
        self._prepare_data()

    def _prepare_data(self):

        # initialise container for splatted data
        self._stats = {k: pd.DataFrame(columns=self._model_description[k]) for k in self._model_description.keys()}

        # fill containers
        for epoch in self._raw_stats:
            for layer, vals in epoch.items():
                self._stats[layer] = self._stats[layer].append(dict(vals), ignore_index=True)

        # change NaN values to 0 and all values to integers
        for layer, vals in self._stats.items():
            self._stats[layer] = vals.fillna(0).astype(int)

    def __str__(self):
        return str(self._raw_stats)

    def plot(self):
        fig = plt.figure()
        for i, layer in enumerate(self._stats, 1):
            ax = fig.add_subplot(len(self._stats), 1, i)
            self._stats[layer].plot(ax=ax, legend=True)
            ax.set_title(layer)
            ax.legend(loc='upper right')
            ax.grid()
            if i == len(self._stats):
                ax.set_xlabel('Epoch')

        plt.tight_layout(.5)
        plt.show()

