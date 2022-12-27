"""A definition of the seed selector based on k-shell algorithm."""

from typing import Any, List

import networkx as nx

from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import bold_underline, thin_underline


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
            f"{bold_underline}\nseed selection method\n{thin_underline}\n"
            f"\tnodewise K Shell decomposition\n{bold_underline}\n"
        )
