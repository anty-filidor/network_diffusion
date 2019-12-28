import itertools
import networkx as nx
from random import choice
from multipledispatch import dispatch
import numpy as np


class PropagationModel:

    def add(self, layer, type):
        """
        This method adds propagation model for the given layer
        :param layer: (str) name of network's layer
        :param type: (list of strings) type of model, generally names of states, i.e. ['s', 'i', 'r']
        """
        self.__setattr__(layer, type)

    def describe(self, to_print=True, full_graph=False):
        """
        Method to quickly print out parameters of the object
        :param to_print: (bool) flag, if true method prints out description to console, otherwise it returns string
        :param full_graph: (bool) flag, if true metrod prints out all propagation model states with those with 0 weight
        :return:
        """
        s = ''
        o = '============================================\n' \
            'model of propagation\n' \
            '--------------------------------------------\n'
        o += 'phenomenas and their states:'
        for name, val in self.__dict__.items():
            if name is 'graph':
                s += '\n'
                for g_name, g in val.items():
                    if full_graph:
                        s += '\n{} transitions: {}\n'.format(g_name, g.edges().data())
                    else:
                        s += '\nlayer \'{}\' transitions with nonzero probability:\n'.format(g_name)
                        for edge in g.edges():
                            if g.edges[edge]['weight'] != 0.0:
                                mask = [True if g_name in _ else False for _ in edge[0]]
                                start = np.array(edge[0])[mask][0].split('.')[1]
                                finish = np.array(edge[1])[mask][0].split('.')[1]
                                constraints = np.array(edge[0])[[not _ for _ in mask]]
                                weight = g.edges[edge]['weight']
                                s += '\tfrom {} to {} with probability {} and constrains {}\n'.\
                                    format(start, finish, weight, constraints)
            else:
                o += '\n\t{}: {}'.format(name, val)

        s += '============================================'
        if to_print:
            print(o, s)
        else:
            return o + s

    def get_model_hyperparams(self):
        """
        Auxiliary method to get model hyperparameters, that is names of 'layers' and states in each layer
        :return: dictionary; key (string) is a name of layer and values are tuples of states labels
        """
        hyperparams = {}
        for name, val in self.__dict__.items():
            if name is not 'graph':
                hyperparams[name] = val
        return hyperparams

    def compile(self, track_changes=False):
        """
        This method creates transition matrices for defined models of propagation in each layer. All transitions
        probabilities are set to 0. To be more specific, transitions matrices are stored as a networkx one-directional
        graph. After compilation user is able to set certain transitions in model
        :param track_changes: (boolean) flag to track progress of matrices creation
        """
        # initialise dictionary to store transition graphs for each later
        transitions_graphs = {}

        # create transition graph for each layer
        for layer, _ in self.__dict__.items():
            if track_changes: print(layer)

            # prepare names in current layer
            cl_names = [str(layer) + '.' + str(v) for v in self.__dict__[layer]]
            if track_changes: print('crtlr', cl_names)

            # copy layers description and delete current layer to prepare constants for current layer
            self_dict_copy = self.__dict__.copy()
            del self_dict_copy[layer]

            # prepare names in other layers which are constant to current layer
            ol_names = []
            for i in self_dict_copy.keys():
                i_names = [str(i) + '.' + str(v) for v in self_dict_copy[i]]
                ol_names.append(i_names)

            # prepare constants
            product = [*itertools.product(*ol_names)]
            if track_changes: print('const', product)

            # crate transitions graph from product and names of states in current layer
            graph = nx.DiGraph()
            for p in product:
                transitions = [tuple(sorted((*a, b))) for a, b in [*itertools.product([p], cl_names)]]
                for edge in itertools.permutations(transitions, 2):
                    if track_changes: print(edge)
                    graph.add_edge(*edge, weight=0.0)
            transitions_graphs.update({layer: graph})

        # save created graphs as attribute of the object
        self.graph = transitions_graphs

    @dispatch(str, tuple, object)
    def set_transition(self, layer, transition, weight):
        """
        This method sets (activate) weight of certain transition in propagation model
        :param layer: (str) name of the later in model
        :param transition: (tuple of strings) name of transition to be activated
        :param weight: in range (number [0, 1]) of activation
        """
        assert 1 >= weight >= 0, 'Weight value should be in [0, 1] range'
        self.graph[layer].edges[transition]['weight'] = weight

    @dispatch(str, str, tuple, object)
    def set_transition(self, initial_layer_attribute, final_layer_attribute, constraint_attributes, weight):
        """
        This method sets (activate) weight of certain transition in model                                                                                                                                    ain transition in propagation model
        :param initial_layer_attribute: (str) value of initial attribute which is being transited
        :param final_layer_attribute: (str) value of final attribute which is being transition
        :param constraint_attributes: (tuple) other attributes available in the propagation model
        :param weight: weight (in range [0, 1]) of activation
        """
        # check data validate
        assert 1 >= weight >= 0, 'Weight value should be in [0, 1] range'
        layer_name = initial_layer_attribute.split('.')[0]
        assert layer_name == final_layer_attribute.split('.')[0], 'Layers must  have one name!'

        # make transitions
        initial_state = tuple(sorted([initial_layer_attribute, *constraint_attributes]))
        final_state = tuple(sorted([final_layer_attribute, *constraint_attributes]))
        self.graph[layer_name].edges[(initial_state, final_state)]['weight'] = weight

    def set_transitions_in_random_edges(self, weights):
        """
        This method sets out random transitions in propagation model using given weights
        :param weights: list of weights to be set in random nodes e.g. for model of 3 layers that list [[0.1, 0.2],
                        [0.03, 0.45], [0.55]] will change 2 weights in first layer, 2, i second and 1 in third
        """
        # check if given probabilities list is the same as number of layers
        assert len(weights) == len(self.graph), 'Len probabilities list and number of layers in model ' \
                                                'should be the same!'

        # main loop
        for (name, graph), wght in zip(self.graph.items(), weights):

            edges_list = []
            for w in wght:

                # select random edge and check if it has not been selected previously
                edge = choice([*graph.edges()])
                while edge in edges_list:
                    edge = choice([*graph.edges()])
                edges_list.append(edge)

                # assign weight to picked edge
                self.set_transition(name, edge, w)

    def get_possible_transitions(self, state, layer):
        """
        Method which returns possible (with weights > 0) transitions from given state in given layer of model
        :param state: (tuple) state of the propagation model, i.e. ('awareness.UA', 'illness.I', 'vaccination.V')
        :param layer: (str) name of the layer of propagation model from which possible transitions are being returned
        :return: list with possible transitions in shape of:
                 [[possible different state in given layer, weight], [], ..., []]
        """

        # check if model has been compiled
        assert self.__dict__['graph'] is not None, 'Failed to process. Compile model first!'

        # read possible states
        states = {}
        for neighbour in self.graph[layer].neighbors(state):
            if self.graph[layer].edges[(state, neighbour)]['weight'] > 0:
                # parse general state to keep only state name in given layer
                for n in neighbour:
                    if layer in n:
                        states[n.split('.')[1]] = self.graph[layer].edges[(state, neighbour)]['weight']
        # print('Possible transitions from state {} in layer {}:\n{}'.format(state, layer, states))
        return states

    #TODO - write method which will add noisy transition in all edges


'''
model = PropagationModel()
model.add('layer_1', ('A', 'B'))
model.add('layer_2', ('A', 'B'))
model.add('layer_3', ('A', 'B', 'C'))
model.compile()
model.describe(full_graph=True)
model.set_transition('layer_1', (('layer_1.A', 'layer_2.A', 'layer_3.A'), ('layer_1.B', 'layer_2.A', 'layer_3.A')), 0.5)
model.set_transitions_in_random_edges([[0.2, 0.3, 0.4], [0.2], [0.3]])
model.describe()
'''
