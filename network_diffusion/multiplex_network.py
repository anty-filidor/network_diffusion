from network_diffusion.multilayer_abstract_network import MultilayerAbstractNetwork
from network_diffusion.utils import read_mlx


class MultiplexNetwork(MultilayerAbstractNetwork):

    def __init__(self):
        self.layers = {}
        self.models = None

    def load_ntework_mlx(self):
        pass

    def load_layers_nx(self, network_list, metadata_list=None):
        """
        This function loads multilayer network as list of layers (nx. networks) and list of its labels
        :param network_list: list of nx networks
        :param metadata_list: list of metadata. It can be none, then labels are set automatically
        """
        if metadata_list is not None:
            assert len(network_list) == len(metadata_list), 'Length of network list and metadata list is not equal'
            for layer, name in zip(network_list, metadata_list):
                self.layers.update({name: layer})
        else:
            for i, layer in enumerate(network_list, start=0):
                name = 'layer_{}'.format(i)
                self.layers.update({name: layer})

    def load_from_one_layer_nx(self, layer, metadata_list=None):
        pass

    def iteration_nodes(self):
        for name, layer in self.layers.items():
            yield(name)
            for n in layer.nodes():
                print(n)
                yield(n)

    def iteration_edges(self, stop_condition):
        pass


















import networkx as nx
M = [nx.les_miserables_graph(), nx.les_miserables_graph(), nx.les_miserables_graph()]

mpx = MultiplexNetwork()

mpx.load_layers_nx(M, ['layer1', 'layer2', 'layer3'])
print(mpx.layers)
mpx.layers = {}

mpx.load_layers_nx(M)
print(mpx.layers)

print(*mpx.layers.keys())