from random import shuffle
from tqdm import tqdm
from copy import deepcopy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from network_diffusion.utils import create_directory

def foo(off):
    """
    jsd;lfjsel;gfkds;lgfjk;ldfj
    :param off:
    :return:
    """
    return  off



class MultiSpreading:
    """
    aaa kotki dwaa
    """

    def __init__(self, model, network):
        """
        This class is to perform experiments defined by Model class on given network
        :param model: (PropagationModel) model of propagation which determines how experiment looks like
        :param network: (MultiplexNetwork) a network which is being examined during experiment
        """

        assert network.layers.keys() == model.get_model_hyperparams().keys(), \
            'Layer names in network should be the same as layer names in propagation model'
        self._model = model
        self._network = deepcopy(network)

    #TODO - change states to be defined as percentages not absolute nodes amounts
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
        logger = ExperimentLogger(self._model.describe(to_print=False), self._network.describe(to_print=False))

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

        # convert logs to dataframe
        logger.convert_logs(self._model.get_model_hyperparams())

        return logger


class ExperimentLogger:
    def __init__(self, model_description, network_description):
        """
        This class is to store and process logs aggregated during propagation experiment of MultiSpreading class
        :param model_description: (str) a description of the model (PropagationModel.describe()) which is used for
                                  saving in logs
        :param network_description: (str) z description of the network (MultiplexNetwork.describe()) which is used for
                                    saving in logs
        """

        self._model_description = model_description
        self._network_description = network_description
        self._raw_stats = []
        self._stats = None

    def add_log(self, log):
        """
        Method which adds raw log from single epoch to the object
        :param log: (dict) raw log - MultiplexNetwork.get_nodes_states()
        """

        self._raw_stats.append(log)

    def convert_logs(self, model_hyperparameters):
        """
        Method to convert raw logs into pandas dataframe. Used after finishing aggregation of logs. It fulfill
        self._stats attribute
        :param model_hyperparameters: (dict) parameters of the propagation model to store
        """

        # initialise container for splatted data
        self._stats = {k: pd.DataFrame(columns=model_hyperparameters[k]) for k in model_hyperparameters.keys()}

        # fill containers
        for epoch in self._raw_stats:
            for layer, vals in epoch.items():
                self._stats[layer] = self._stats[layer].append(dict(vals), ignore_index=True)

        # change NaN values to 0 and all values to integers
        for layer, vals in self._stats.items():
            self._stats[layer] = vals.fillna(0).astype(int)

    def __str__(self):
        """
        Method which allows to print out object
        :return: (str) string representing object
        """
        return str(self._stats)

    def plot(self, to_file=False, path=None):
        """
        Method to plot out visualisation of performed experiment
        :param to_file: (bool) flag, if true save figure to file, otherwise it is plotted on screen
        :param path: (str) path to save figure
        """

        fig = plt.figure()

        for i, layer in enumerate(self._stats, 1):
            ax = fig.add_subplot(len(self._stats), 1, i)
            self._stats[layer].plot(ax=ax, legend=True)
            ax.set_title(layer)
            ax.legend(loc='upper right')
            ax.set_ylabel('Nodes')
            if i == 1:
                nn = self._stats[layer].iloc[0].sum()
            ax.set_yticks(np.arange(0, nn+1, 20))
            if i == len(self._stats):
                ax.set_xlabel('Epoch')
            ax.grid()

        plt.tight_layout(.3)
        if to_file:
            plt.savefig("{}/visualisation.png".format(path), dpi=200)
        else:
            plt.show()

    def report(self, visualisation=False, to_file=False, path=None):
        """
        Method to create report of experiment. It consits of report of the network, report of the model, record of
        propagation progress and optionally visualisation of the progress.
        :param to_file: (bool) a flag, if true report is saved in files, otherwise it is printed out on the screen
        :param path: (str) path to folder where report will be saved
        :param visualisation: (bool) a flag, if true visualisation is being plotted
        """

        if to_file:
            # create directory from given path
            create_directory(path)
            # save progress in propagation of each layer to csv file
            for stat in self._stats:
                self._stats[stat].to_csv(path+'/'+stat+'_propagation_report.csv', index_label='epoch')
            # save description of model to txt file
            with open(path+'/model_report.txt', 'w') as file:
                file.write(self._model_description)
            # save description of network to txt file
            with open(path+'/network_report.txt', 'w') as file:
                file.write(self._network_description)
            # save figure
            if visualisation:
                self.plot(to_file=True, path=path)
        else:
            print(self._network_description)
            print(self._model_description)
            print('============================================\n'
                  'propagation report\n'
                  '--------------------------------------------')
            for stat in self._stats:
                print(stat, '\n', self._stats[stat], '\n')
            print('============================================')
            if visualisation:
                self.plot()
