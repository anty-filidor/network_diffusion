from network_diffusion.multilayer_abstract_network import MultilayerAbstractNetwork
from network_diffusion.utils import read_mlx
import networkx as nx
from collections import Counter
import numpy as np


class MultiplexNetwork(MultilayerAbstractNetwork):

    def __init__(self):
        self.layers = {}

    def describe(self, to_print=True):
        """
        Method to quickly print out parameters of the object
        """
        assert len(self.layers) > 0, 'Import network to the object first!'
        s = '============================================\n' \
            'network parameters\n' \
            '--------------------------------------------\n'

        s += 'general parameters:\n'
        s += '\tnumber of layers: {}\n'.format(len(self.layers))
        mul_coeff = self._compute_multiplexing_coefficient()
        s += '\tmultiplexing coefficient: {}\n'.format(round(mul_coeff, 4))

        for name, graph in self.layers.items():
            s += '\nlayer \'{}\' parameters:\n'.format(name)
            s += '\tgraph type - {}\n\tnumber of nodes - {}\n\tnumber of edges - {}\n'.\
                format(type(graph), len(graph.nodes()), len(graph.edges()))
            avg_deg = np.average([_[1] for _ in graph.degree()])
            s += '\taverage degree - {}\n'.format(round(avg_deg, 4))
            s += '\tclustering coefficient - {}\n'.format(round(nx.average_clustering(graph), 4))
            nx.draw(graph)
        s += '============================================'

        if to_print:
            print(s)
        else:
            return s

    def _compute_multiplexing_coefficient(self):
        """
        Method to compute multiplexing coefficient defined as proportion of number of nodes common to all layers to
        number of all unique nodes in entire network
        :return: (float) multiplexing coefficient
        """

        assert len(self.layers) > 0, 'Import network to the object first!'

        # create set of all nodes in the network
        all_nodes = set()
        for graph in self.layers.values():
            all_nodes.update([*graph.nodes()])

        # create set of all common nodes in the network over the layers
        common_nodes = set()
        for node in all_nodes:
            _ = True
            for l in self.layers.values():
                if node not in l:
                    _ = False
                    break
            if _ is True:
                common_nodes.add(node)

        return len(common_nodes) / len(all_nodes)


    def get_nodes_states(self):
        """
        Method which returns summary of network's nodes states.
        :return: dictionary with items representing each of layers and with summary of node states in values
        """
        assert len(self.layers) > 0, 'Import network to the object first!'
        statistics = {}
        for name, layer in self.layers.items():
            tab = []
            for node in layer.nodes():
                tab.append(layer.node[node]['status'])
            statistics[name] = tuple(Counter(tab).items())
        return statistics

    def get_node_state(self, node):
        """
        Method which returns general state of node in network. In format acceptable by PropagationModel object, which
        means a tuple of sorted strings in form <layer name>.<state>
        :param node: a node which status has to be returned
        :return: dictionary with layer names as keys and states of node in values
        """
        assert len(self.layers) > 0, 'Import network to the object first!'
        statistics = []
        for name, layer in self.layers.items():
            statistics.append('{}.{}'.format(name, layer.node[node]['status']))
        statistics = sorted(statistics)
        return tuple(statistics)

    def _prepare_nodes_attribute(self):
        for layer in self.layers.values():
            status_dict = {n: None for n in layer.nodes()}
            nx.set_node_attributes(layer, status_dict, 'status')

    # TODO
    def load_ntework_mlx(self):
        """
        To do !!!!!!
        :return:
        """
        pass

    def load_layers_nx(self, network_list, layer_names=None):
        """
        This function loads multilayer network as list of layers (nx. networks) and list of its labels
        :param network_list: list of nx networks
        :param layer_names: list of layer names. It can be none, then labels are set automatically
        """
        if layer_names is not None:
            assert len(network_list) == len(layer_names), 'Length of network list and metadata list is not equal'
            for layer, name in zip(network_list, layer_names):
                self.layers.update({name: layer})
        else:
            for i, layer in enumerate(network_list, start=0):
                name = 'layer_{}'.format(i)
                self.layers.update({name: layer})

        self._prepare_nodes_attribute()

    # TODO
    def load_from_one_layer_nx(self, layer, metadata_list=None):
        """
        To do !!!!!!
        :param layer:
        :param metadata_list:
        :return:
        """
        pass

    def iterate_nodes_layer(self, layer):
        for n in self.layers[layer].nodes():
            yield(n)

    def iterate_edges_layer(self, layer):
        for e in self.layers[layer].nodes():
            yield(e)

    def get_neighbours_attributes(self, node, layer):
        assert len(self.layers) > 0, 'Import network to the object first!'
        attributes = []
        neighbours = [*self.layers[layer].neighbors(node)]
        for neighbour in neighbours:
             attributes.append(self.layers[layer].node[neighbour])
        return attributes


'''
M = [nx.les_miserables_graph(), nx.les_miserables_graph(), nx.les_miserables_graph()]

mpx = MultiplexNetwork()
mpx.load_layers_nx(M, ['layer1', 'layer2', 'layer3'])
print(mpx.layers)

mpx.layers = {}
mpx.load_layers_nx(M)
print(mpx.layers)

for n in mpx.iterate_nodes_layer('layer_1'):
    print(mpx.get_neighbours_attributes(n, 'layer_1'))
'''
