from abc import ABC, abstractmethod

# https://drive.google.com/drive/folders/0B_M5Zh3gg4LkNWZ5WmpoRlJhMVE

#TODO Consider delete it

#TYPE
#LAYERS
#ACTOR ATTRIBUTES
#NODE ATTRIBUTES
#EDGE ATTRIBUTES
#ACTORS
#NODES
#EDGES

class MultilayerAbstractNetwork(ABC):

    @abstractmethod
    def iterate_nodes_layer(self, layer):
        pass

    @abstractmethod
    def iterate_edges_layer(self, layer):
        pass
