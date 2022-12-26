"""Randomised seed selector for DSAA algorithm."""
from random import shuffle
from typing import Any, List

import networkx as nx

from network_diffusion.seeding.base_selector import BaseSeedSelector


class RandomSeedSelector(BaseSeedSelector):
    """Randomised seed selector for DSAA algorithm."""

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a random ranking of nodes.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        nodes_list = [*graph.nodes()]
        shuffle(nodes_list)
        return nodes_list
