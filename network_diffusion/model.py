import numpy as np
import itertools
import networkx as nx


class Model:

    def add(self, layer, type):
        """
        This method adds propagation model for the given layer
        :param layer: (str) name of network's layer
        :param type: (table of string) type of model, generally names of states, i.e. ['s', 'i', 'r']
        """
        self.__setattr__(layer, type)

    def describe(self):
        """
        Method to quickly check out parameters of the object
        :return:
        """
        for name, val in self.__dict__.items():
            if name is 'graph':
                print('\n')
                for n, v in val.items():
                    print(n, 'transitions:', v.edges().data())
            else:
                print(name, ':', val)
        print('\n')

    def compile(self, track_changes=False):
        """
        This method creates transition matrices for defined models of propagation in each layer. All transitions
        probablilties are set to 0. To be moer specific, transitions matrices are stored as a networkx one-directional
        graph. After compilation user is able to set certain transitions in model
        :param track_changes: (boolean) flag to track progress of matrices creation
        :return:
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

    def set_transition(self, transition):
        pass




model = Model()
model.add('layer_0', ('A', 'B'))
model.add('layer_1', ('A', 'B'))
model.add('layer_2', ('A', 'B'))
model.add('layer_3', ('A', 'B', 'C'))
model.describe()
model.compile()
model.describe()