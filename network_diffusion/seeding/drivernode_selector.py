"""A definition of the seed selector based on driver nodes."""

from typing import Any, List

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.seeding.betweenness_selector import BetweennessSelector
from network_diffusion.seeding.closeness_selector import ClosenessSelector
from network_diffusion.seeding.degreecentrality_selector import (
    DegreeCentralitySelector,
)
from network_diffusion.seeding.katz_selector import KatzSelector
from network_diffusion.seeding.kshell_selector import KShellMLNSeedSelector
from network_diffusion.seeding.pagerank_selector import PageRankMLNSeedSelector
from network_diffusion.seeding.random_selector import RandomSeedSelector
from network_diffusion.seeding.voterank_selector import VoteRankSeedSelector
from network_diffusion.tpn.functions import compute_driver_nodes
from network_diffusion.tpn.tpnetwork import TemporalNetwork
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class DriverNodeSelector(BaseSeedSelector):
    """Driver Node based seed selector."""

    def __init__(self, method: str) -> None:
        """Initialise object."""
        match method:
            case "random":
                self.selector = RandomSeedSelector()
            case "degree":
                self.selector = DegreeCentralitySelector()
            case "closeness":
                self.selector = ClosenessSelector()
            case "betweenness":
                self.selector = BetweennessSelector()
            case "katz":
                self.selector = KatzSelector()
            case "kshell":
                self.selector = KShellMLNSeedSelector()
            case "pagerank":
                self.selector = PageRankMLNSeedSelector()
            case "voterank":
                self.selector = VoteRankSeedSelector()
            case _:
                self.selector = None
        super().__init__()

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tdriver node\n{BOLD_UNDERLINE}\n"
        )

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    # TODO: other methods and parameters?
    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Return a list of driver actors for a multilayer network."""
        driver_nodes_list = compute_driver_nodes(net)

        if self.selector is None:
            return self._reorder_seeds(driver_nodes_list, net.get_actors())

        result = self.selector.actorwise(net)
        result = self._reorder_seeds(driver_nodes_list, result)
        return result

    def snap_select(
        self, net: TemporalNetwork, snap_id: int
    ) -> List[MLNetworkActor]:
        """Return a list of driver actors for a temporal network."""
        snap = net.snaps[snap_id]
        return self.actorwise(snap)

    @staticmethod
    def _reorder_seeds(
        driver_nodes: List[MLNetworkActor],
        all_actors: List[MLNetworkActor],
    ):
        """Return a list of node ids, where driver nodes in the first."""
        result = []

        driver_ac_set = set(driver_nodes)

        for item in all_actors:
            if item in driver_ac_set:
                result.append(item)
                all_actors.remove(item)

        result.extend(all_actors)
        return result
