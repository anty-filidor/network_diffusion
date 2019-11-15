from network_diffusion.multilayer_abstract_network import MultilayerAbstractNetwork
from network_diffusion.utils import read_mlx


class MultLayerNetwork(MultilayerAbstractNetwork):

    def __init__(self):
        self.layers = {}

    def load_ntework_mlx(self):
        pass

    def load_network_nx(self, network_list, metadata_list=None):
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

    def iteration_nodes(self, stop_condition):
        pass
