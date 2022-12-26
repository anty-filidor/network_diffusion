"""A definition of the base seed selector class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from network_diffusion.multilayer_network import MultilayerNetwork


class BaseSeedSelector(ABC):
    """Base abstract class for seed selectors."""

    @abstractmethod
    def __call__(self, network: MultilayerNetwork) -> Dict[str, List[Any]]:
        """
        Prepare ranking list according to implementing seeding strategy.

        :param network: a network to calculate ranking list for
        :return: a dictionary keyed by layer names with lists of node names
            ordered in descending way according to seed selection strategy
        """
        ...
