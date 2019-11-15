from abc import ABC, abstractmethod

# https://drive.google.com/drive/folders/0B_M5Zh3gg4LkNWZ5WmpoRlJhMVE


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
    def iteration_nodes(self):
        pass

    @abstractmethod
    def iteration_edges(self):
        pass
