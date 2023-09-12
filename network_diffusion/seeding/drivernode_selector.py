"""A definition of the seed selector based on driver nodes."""

from typing import List

# from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.tpn.tpnetwork import TemporalNetwork
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.mln.actor import MLNetworkActor

from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE

from network_diffusion.tpn.functions import driver_nodes
from network_diffusion.seeding.random_selector import RandomSeedSelector
from network_diffusion.seeding.betweenness_selector import BetweennessSelector
from network_diffusion.seeding.degreecentrality_selector import DegreeCentralitySelector
from network_diffusion.seeding.closeness_selector import ClosenessSelector
from network_diffusion.seeding.katz_selector import KatzSelector
from network_diffusion.seeding.kshell_selector import KShellMLNSeedSelector
from network_diffusion.seeding.mocky_selector import MockyActorSelector
from network_diffusion.seeding.neighbourhoodsize_selector import NeighbourhoodSizeSelector
from network_diffusion.seeding.pagerank_selector import PageRankMLNSeedSelector
from network_diffusion.seeding.voterank_selector import VoteRankSeedSelector
from network_diffusion.seeding.cbim import CBIMselector

class DriverNodeSelector():
    """Driver Node based seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tdriver node\n{BOLD_UNDERLINE}\n"
        )
    
    #TODO: other methods and parameters?
    def snap_select(self, net: TemporalNetwork, snap_id: int, method: str) -> List[int]:
        """Return list of driver node"""

        snap = net.snaps[snap_id]

        driver_nodes_list = driver_nodes(snap)

        match method:
            case "random":
                selector = RandomSeedSelector()
            case "degree":
                selector = DegreeCentralitySelector()
            case "closeness":
                selector = ClosenessSelector()         
            case "betweenness":
                selector = BetweennessSelector()
            case "katz":
                selector = KatzSelector()
            case "kshell":
                selector = KShellMLNSeedSelector()
            case "pagerank":
                selector = PageRankMLNSeedSelector()
            case "voterank":
                selector = VoteRankSeedSelector()
            case _:
                return self.reorder_seeds(driver_nodes_list, snap.get_actors())

        result = selector.actorwise(snap)
        result = self.reorder_seeds(driver_nodes_list, result)
        return result
        
    #TODO: keeping the rest non-driver-nodes or not?
    def reorder_seeds(self, driver_nodes: List[int], all_nodes: List[MLNetworkActor]):
        """Return a list of node ids, where driver nodes in the first"""
        result = []
        all_nodes = [x.actor_id for x in all_nodes]

        for item in all_nodes[:]:
            if item in driver_nodes:
                result.append(item)
                all_nodes.remove(item)

        # result.extend(all_nodes)
        return result



