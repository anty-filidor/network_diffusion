"""A definition of the seed selector based on k-shell algorithm."""

from itertools import zip_longest
from random import shuffle
from typing import Any, List

import networkx as nx

from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE
from  network_diffusion.mln.mln_actor import MLNetworkActor
from network_diffusion.mln.mln_network import MultilayerNetwork

class KShellSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on k-shell algorithm."""

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
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

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tnodewise K Shell decomposition\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """COmpute ranking for actors."""
        actor_ranking, picked_actors = [], []
        for nth_nodes in zip_longest(*self.nodewise(net).values(), fillvalue=None):
            shuffle(list(nth_nodes))
            for node_id in nth_nodes:
                if node_id is None or node_id in picked_actors:
                    continue
                actor_ranking.append(net.get_actor(actor_id=node_id))
                picked_actors.append(node_id)
        assert len(picked_actors) == net.get_actors_num(), "Incorrect ranking!"
        return actor_ranking