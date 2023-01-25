"""A definition of the seed selectors based on k-shell algorithm."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import degree, k_shell_mln
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import (
    BaseSeedSelector,
    node_to_actor_ranking,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class KShellSeedSelector(BaseSeedSelector):
    """
    Selector for MLTModel based on k-shell algorithm.

    According to "Seed selection for information cascade in multilayer
    networks" by Fredrik Erlandsson, Piotr Bródka, and Anton Borg we have
    extended k-shell ranking by combining it with degree of the node in each
    layer, so that ranking is better ordered (nodes in shells can be ordered).
    """

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """
        Create a ranking based on nodes' k-shell cohort position & degree.

        :param graph: single layer graph to compute ranking for
        :return: list of node-ids ordered descending by their ranking position
        """
        ksh_deepest_nodes = set(nx.k_shell(graph).nodes())
        shell_ranking = {}
        k = 0

        # iterate until deepest shell is achieved
        while True:

            # compute k-shell cohort
            ksh_nodes = [*nx.k_shell(graph, k=k).nodes()]

            # sort it according to degree in the graph
            shell_ranking[k] = sorted(
                ksh_nodes,
                key=lambda x: nx.degree(graph)[x],  # type: ignore
                reverse=True,
            )

            # if the deepest shell is reached breake, othrwise increase k
            if set(ksh_nodes) == ksh_deepest_nodes:
                break
            k += 1

        # flatten ranking to the ordered list
        return [
            node
            for cohort in sorted(shell_ranking)[::-1]
            for node in shell_ranking[cohort]
        ]

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tK Shell decomposition computed nodewise\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Compute ranking for actors."""
        return node_to_actor_ranking(super().nodewise(net), net)


class KShellMLNSeedSelector(BaseSeedSelector):
    """
    Selector for MLTModel based on k-shell algorithm.

    In contrary to KShellSeedSelector it utilises k-shell decomposition defined
    as in network_diffusion.mln.functions.k_shell_mln()
    """

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tK Shell decomposition computed actorwise\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Compute ranking for actors."""
        ksh_deepest_actors = set(k_shell_mln(net=net).get_actors())
        shell_ranking = {}
        k = 0

        # iterate until deepest shell is achieved
        while True:

            # compute k-shell cohort
            ksh_actors = k_shell_mln(net=net, k=k).get_actors()

            # sort it according to degree in the graph
            shell_ranking[k] = sorted(
                ksh_actors,
                key=lambda x: degree(net)[x],
                reverse=True,
            )

            # if the deepest shell is reached breake, othrwise increase k
            if set(ksh_actors) == ksh_deepest_actors:
                break
            k += 1

        # flatten ranking to the ordered list
        return [
            node
            for cohort in sorted(shell_ranking)[::-1]
            for node in shell_ranking[cohort]
        ]
