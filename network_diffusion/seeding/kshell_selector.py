"""A definition of the seed selector based on k-shell algorithm."""

from typing import Any, Dict, List

from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.multilayer_network import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector

import networkx as nx

class KShellSelectorMLTM(BaseSeedSelector):
    """Selector for MLTModel based on k-shell algorithm."""

    def __call__(self, network: MultilayerNetwork) -> Dict[str, Any]:
        """Prepare ranking list for each layer."""
        nodes_ranking = {}
        for l_name, l_graph in network.layers.items():
            seeds_in_layer = self._create_k_shell_ranking_for_layer(l_graph)
            nodes_ranking[l_name] = seeds_in_layer
        return nodes_ranking

    @staticmethod
    def _create_k_shell_ranking_for_layer(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes based on their k-shell cohort position.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ksh_deepest_nodes = set(nx.k_shell(graph).nodes())
        shell_ranking = {}
        k = 0

        while True:
            ksh_nodes = set(nx.k_shell(graph, k=k).nodes())
            shell_ranking[k] = ksh_nodes
            if ksh_nodes == ksh_deepest_nodes:
                break
            k += 1
        
        return [
            node 
            for cohort in sorted(shell_ranking)[::-1] 
            for node in shell_ranking[cohort]
        ]
