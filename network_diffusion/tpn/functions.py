
from typing import List

import networkx as nx
from networkx.algorithms.approximation.dominating_set import min_weighted_dominating_set
from networkx.algorithms.dominating import dominating_set


from network_diffusion.tpn.tpnetwork import TemporalNetwork

def driver_nodes(net: nx.Graph) -> List[int]:
    """Return driver nodes for a given network"""

    # min_dominating_set = min_weighted_dominating_set(net)
    min_dominating_set = dominating_set(net)


    return min_dominating_set




