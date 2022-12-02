"""A definition of the base seed selector class."""

from abc import ABC
from typing import List

from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.multilayer_network import MultilayerNetwork


class BaseSeedSelector(ABC):
    """Base abstract class for seed selectors."""

    def __call__(
        self, model: CompartmentalGraph, network: MultilayerNetwork
    ) -> List[NetworkUpdateBuffer]:
        """Prepare ranking list according to implementing seeding strategy."""
        raise NotImplementedError
