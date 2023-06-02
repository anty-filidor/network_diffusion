"""A definition of the seed selector based on Katz centrality."""

from typing import List

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.functions import katz
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class KatzSelector(BaseSeedSelector):
    """Katz Centrality seed selector."""

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tkatz centrality choice\n{BOLD_UNDERLINE}\n"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking for actors using Katz centrality metric."""
        ranking_list: List[MLNetworkActor] = []
        # sorted by value of dictonaries
        for dc_val, _ in sorted(
            katz(net=net).items(), reverse=True, key=lambda x: x[1]
        ):
            # Adding actor to list
            ranking_list.append(dc_val)
        return ranking_list
