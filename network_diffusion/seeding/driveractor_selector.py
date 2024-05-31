"""A definition of the seed selector based on driver actors."""

from typing import Any, List, Optional

import networkx as nx

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.driver_actors import compute_driver_actors
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.tpn.tpnetwork import TemporalNetwork
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class DriverActorSelector(BaseSeedSelector):
    """Driver Actor based seed selector."""

    def __init__(self, method: Optional[BaseSeedSelector]) -> None:
        """
        Initialise object.

        :param method: a method to sort driver actors.
        """
        super().__init__()
        if isinstance(method, DriverActorSelector):
            raise AttributeError(
                f"Argument 'method' cannot be {self.__class__.__name__}!"
            )
        self.selector = method

    def __str__(self) -> str:
        """Return seed method's description."""
        selector = self.selector
        s_str = "" if selector is None else f" + {selector.__class__.__name__}"
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tdriver actor{s_str}\n{BOLD_UNDERLINE}\n"
        )

    def _calculate_ranking_list(self, graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    # TODO: other methods and parameters?
    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Return a list of driver actors for a multilayer network."""
        driver_actors_list = compute_driver_actors(net)

        if self.selector is None:
            return self._reorder_seeds(driver_actors_list, net.get_actors())

        result = self.selector.actorwise(net)
        result = self._reorder_seeds(driver_actors_list, result)
        return result

    def snap_select(
        self, net: TemporalNetwork, snap_id: int
    ) -> List[MLNetworkActor]:
        """Return a list of driver actors for a temporal network."""
        snap = net.snaps[snap_id]
        return self.actorwise(snap)

    @staticmethod
    def _reorder_seeds(
        driver_actors: List[MLNetworkActor],
        all_actors: List[MLNetworkActor],
    ) -> List[MLNetworkActor]:
        """Return a list of actor ids, where driver actors in the first."""
        result = []

        driver_ac_set = set(driver_actors)

        result.extend([item for item in all_actors if item in driver_ac_set])
        result.extend(
            [item for item in all_actors if item not in driver_ac_set]
        )

        return result
