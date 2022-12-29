"""A definition of the seed selector based on Vote Rank algorithm."""

from typing import Any, List

import networkx as nx

from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class VoteRankSeedSelector(BaseSeedSelector):
    """Selector for MLTModel based on Vote Rank algorithm."""

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking of nodes with Vote Rank algorithm.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        return nx.voterank(graph)

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tnodewise Voterank\n{BOLD_UNDERLINE}\n"
        )
