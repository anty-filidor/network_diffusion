from network_diffusion.utils import read_mlx
import networkx as nx
from collections import Counter
import numpy as np
from copy import deepcopy


class MultilayerNetwork:

    def __init__(self):
        """
        Init method, creates empty object which has to be fullfill by loading a network
        """

        self.layers = {}

    def load_mlx(self, file_path):
        """
        This function loads multilayer network saved in mlx file. Note, that is omits some nonimportant attributes of
        network defined in the file, i.e. node attributes.
        :param file_path: path to the file
        """

        # clear former layers of the object
        self.layers = {}

        # read mlx file
        # print('reading raw file')
        raw_data = read_mlx(file_path)

        # create layers
        # print('creating layers')
        if 'layers' in raw_data.keys():
            for n, t in raw_data['layers']:
                if 'DIRECTED' in t:
                    self.layers[n] = nx.DiGraph()
                else:
                    print('unrecognised layer')
        else:
            raise ResourceWarning('file corrupted - no layers defined')
        '''
        # import nodes
        print('importing nodes')
        if 'nodes' in raw_data.keys():
            for layer in self.layers.keys():
                for node in raw_data['nodes']:
                    self.layers[layer].add_node(node[0])
        '''
        # import edges
        # print('importing edges')
        if 'edges' in raw_data.keys():
            for edge in raw_data['edges']:
                # if edge definition is not corrupted go into it
                if len(edge) >= 3 and edge[2] in self.layers.keys():
                    self.layers[edge[2]].add_edge(edge[0], edge[1])

        self._prepare_nodes_attribute()

    def load_layers_nx(self, network_list, layer_names=None):
        """
        This function loads multilayer network as list of layers (nx. networks) and list of its labels
        :param network_list: list of nx networks
        :param layer_names: list of layer names. It can be none, then labels are set automatically
        """

        # clear former layers of the object
        self.layers = {}

        if layer_names is not None:
            assert len(network_list) == len(layer_names), 'Length of network list and metadata list is not equal'
            for layer, name in zip(network_list, layer_names):
                self.layers.update({name: layer})
        else:
            for i, layer in enumerate(network_list, start=0):
                name = 'layer_{}'.format(i)
                self.layers.update({name: layer})

        self._prepare_nodes_attribute()

    def load_layer_nx(self, network_layer, layer_names):
        """
        This function creates multiplex network from layers defined in layer_names and network_layer which is replicated
        through all layers
        :param network_layer: (networkx.Graph) basic layer which is replicated througa all layers
        :param layer_names: (list of iterables) names for layers in multiplex network
        """

        # clear former layers of the object
        self.layers = {}

        for name in layer_names:
            self.layers.update({name: deepcopy(network_layer)})

        self._prepare_nodes_attribute()

    def _prepare_nodes_attribute(self):
        """
        This method prepares network to experiment by adding an attribute 'status to each node in each layer
        """

        for layer in self.layers.values():
            status_dict = {n: None for n in layer.nodes()}
            nx.set_node_attributes(layer, status_dict, 'status')

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
            if len(graph) > 0:
                s += '\tclustering coefficient - {}\n'.format(round(nx.average_clustering(graph), 4))
            else:
                s += '\tclustering coefficient - nan\n'
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

        if len(all_nodes) > 0:
            return len(common_nodes) / len(all_nodes)
        else:
            return 0

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
            if node in layer:
                statistics.append('{}.{}'.format(name, layer.node[node]['status']))
        statistics = sorted(statistics)
        return tuple(statistics)

    def get_neighbours_attributes(self, node, layer):
        """
        Method which returns attributes in given layer of given node's neighbours
        :param node: name of node
        :param layer: layer from which neighbours are being examined
        :return: list of neighbours' attributes
        """

        assert len(self.layers) > 0, 'Import network to the object first!'
        attributes = []
        neighbours = [*self.layers[layer].neighbors(node)]
        for neighbour in neighbours:
            attributes.append(self.layers[layer].node[neighbour])
        return attributes

    def get_layer_names(self):
        """
        Method which gets names of layers in the network
        :return: list of layers' names
        """

        return [*self.layers.keys()]

    def iterate_nodes_layer(self, layer):
        """
        Generator, yields node by node selected layer of the network
        :param layer: layer from nodes are yield
        :return: node names
        """

        for n in self.layers[layer].nodes():
            yield(n)

    def iterate_edges_layer(self, layer):
        """
        Generator, yields edge by edge selected layer of the network
        :param layer: layer from edges are yield
        :return: edge names
        """

        for e in self.layers[layer].nodes():
            yield(e)


'''
# example of creating net from networkx graphs
M = [nx.les_miserables_graph(), nx.les_miserables_graph(), nx.les_miserables_graph()]

mpx = MultiplexNetwork()
mpx.load_layers_nx(M, ['layer1', 'layer2', 'layer3'])
print(mpx.layers)

mpx.layers = {}
mpx.load_layers_nx(M)
print(mpx.layers)

for n in mpx.iterate_nodes_layer('layer_1'):
    print(mpx.get_neighbours_attributes(n, 'layer_1'))

# example of creating nets from mpx files
nets = ['/Users/michal/PycharmProjects/network_diffusion/network_records/florentine.mpx',
        '/Users/michal/PycharmProjects/network_diffusion/network_records/fftwyt.mpx',
        '/Users/michal/PycharmProjects/network_diffusion/network_records/monastery.mpx']
        # '/Users/michal/PycharmProjects/network_diffusion/network_records/test_bad',
        # '/Users/michal/PycharmProjects/network_diffusion/requirements.txt']

import glob
nets = glob.glob('/Users/michal/PycharmProjects/network_diffusion/network_records/*.mpx')

for net in nets:
    print(net)
    mpx = MultiplexNetwork()
    mpx.load_mlx(net)
    mpx.describe()
'''

