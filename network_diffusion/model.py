import itertools
import networkx as nx
from random import choice


class Model:

    def add(self, layer, type):
        """
        This method adds propagation model for the given layer
        :param layer: (str) name of network's layer
        :param type: (table of string) type of model, generally names of states, i.e. ['s', 'i', 'r']
        """
        self.__setattr__(layer, type)

    def describe(self, full_graph=False):
        """
        Method to quickly print out parameters of the object
        """
        for name, val in self.__dict__.items():
            if name is 'graph':
                print('\n')
                for g_name, g in val.items():
                    if full_graph:
                        print(g_name, 'transitions:', g.edges().data())
                    else:
                        print(g_name, 'nonzero transitions:')
                        for edge in g.edges():
                            if g.edges[edge]['weight'] != 0.0:
                                print(edge, g.edges[edge])
            else:
                print(name, ':', val)
        print('\n')

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

    def set_transition(self, layer, transition, weight):
        """
        This method sets weight (activate) of certain transition in propagation model
        :param layer: (str) name of the later in model
        :param transition: (tuple of strings) name of transition to be activated
        :param weight: weight (number [0, 1]) of activation
        """
        assert 1 >= weight >= 0, 'Weight value should be in [0, 1] range'
        self.graph[layer].edges[transition]['weight'] = weight

    def set_transitions_in_random_edges(self, weights):
        """
        This method sets out random transitions in propagation model using given weights
        :param weights: list of weights to be set in random nodes e.g. for model of 3 layers that list [[0.1, 0.2],
        [0.03, 0.45], [0.55]] will change 2 weights in first layer, 2, i second and 1 in thirs
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



model = Model()
#model.add('layer_0', ('A', 'B'))
model.add('layer_1', ('A', 'B'))
model.add('layer_2', ('A', 'B'))
model.add('layer_3', ('A', 'B', 'C'))
model.compile()
model.describe(full_graph=True)

model.set_transition('layer_1', (('layer_1.A', 'layer_2.A', 'layer_3.A'), ('layer_1.B', 'layer_2.A', 'layer_3.A')), 0.5)
model.set_transitions_in_random_edges([[0.2, 0.3, 0.4], [0.2], [0.3]])
model.describe()