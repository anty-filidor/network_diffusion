"""
Generators of multilayer networks with Preferential Attachment or Erdos-Renyi models.

TODO: use nd based script once it gets updated to the current form!
"""

import abc
from typing import Literal

import numpy as np
import uunet.multinet as ml
from uunet._multinet import PyEvolutionModel, PyMLNetwork

import network_diffusion as nd


class MultilayerBaseGenerator(abc.ABC):
    """Abstract base class for multilayer network generators."""

    @abc.abstractmethod
    def __init__(self, nb_layers: int, nb_actors: int, nb_steps: int) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        """
        self.nb_layers = nb_layers
        self.nb_actors = nb_actors
        self.nb_steps = nb_steps

    @abc.abstractmethod
    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of concrete network generators
        """
        pass

    def get_dependency(self) -> np.ndarray:
        """
        Create a dependency matrix for the network generator.

        :return: matrix shaped: [nb_layers, nb_layers] with 0s in diagonal and 1/nb_layers otherwise
        """
        dep = np.full(
            fill_value=(1 / self.nb_layers),
            shape=(self.nb_layers, self.nb_layers),
        )
        np.fill_diagonal(dep, 0)
        return dep

    def __call__(self) -> PyMLNetwork:
        """
        Generate the network according to given parameters.

        :return: generated network as a PyEvolutionModel object
        """
        models = self.get_models()
        pr_internal = np.random.uniform(0, 1, size=self.nb_layers)
        pr_external = (
            np.ones_like(pr_internal) - pr_internal
        )  # TODO: there is no noaction step now!
        dependency = self.get_dependency()
        return ml.grow(
            self.nb_actors,
            self.nb_steps,
            models,
            pr_internal.tolist(),
            pr_external.tolist(),
            dependency.tolist(),
        )


class MultilayerERGenerator(MultilayerBaseGenerator):
    """Class which generates multilayer network with Erdos-Renyi algorithm."""

    def __init__(
        self, nb_layers: int, nb_actors: int, nb_steps: int, std_nodes: int
    ) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param std_nodes: standard deviation of the number of nodes in each layer (expected value
            is a number of actors)
        """
        super().__init__(
            nb_layers=nb_layers, nb_actors=nb_actors, nb_steps=nb_steps
        )
        assert (
            nb_actors - 2 * std_nodes > 0
        ), "Number of actors shall be g.t. 2 * std_nodes"
        self.std_nodes = std_nodes

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer with num nodes drawn from a standard distribution.

        :return: list of Erdos-Renyi generators
        """
        layer_sizes = (
            np.random.normal(
                loc=self.nb_actors - self.std_nodes,
                scale=self.std_nodes,
                size=self.nb_layers,
            )
            .astype(np.int32)
            .clip(min=1, max=self.nb_actors)
        )
        return [ml.evolution_er(n=lv) for lv in layer_sizes]


class MultilayerPAGenerator(MultilayerBaseGenerator):
    """Class which generates multilayer network with Preferential Attachment algorithm."""

    def __init__(
        self, nb_layers: int, nb_actors: int, nb_steps: int, nb_hubs: int
    ) -> None:
        """
        Initialise the object.

        :param nb_layers: number of layers of the generated network
        :param nb_actors: number of actors in the network
        :param nb_steps: number of steps of the generative algorithm which builds the network
        :param nb_seeds: number of seeds in each layer and a number of egdes from each new vertex
        """
        super().__init__(
            nb_layers=nb_layers, nb_actors=nb_actors, nb_steps=nb_steps
        )
        self.nb_hubs = nb_hubs

    def get_models(self) -> list[PyEvolutionModel]:
        """
        Get evolutionary models for each layer.

        :return: list of Preferential Attachment generators
        """
        m0s = [self.nb_hubs] * self.nb_layers
        ms = m0s.copy()
        return [ml.evolution_pa(m0=m0, m=ms) for m0, ms in zip(m0s, ms)]


def generate(
    model: Literal["PA", "ER"], nb_actors: int, nb_layers: int
) -> None:
    """Generate a random multilayer Erdos-Renyi or Preferential-Attachement network."""
    if model == "ER":
        std_nodes = int(0.1 * nb_actors)
        net_uu = MultilayerERGenerator(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_actors * 10,
            std_nodes=std_nodes,
        )()
    elif model == "PA":
        nb_hubs = int(max(2, 0.05 * nb_actors))
        net_uu = MultilayerPAGenerator(
            nb_layers=nb_layers,
            nb_actors=nb_actors,
            nb_steps=nb_actors * 10,
            nb_hubs=nb_hubs,
        )()
    else:
        raise ValueError("Incorret model name!")

    # convert uunet representation to networkx
    net_nx = ml.to_nx_dict(net_uu)

    # convert networkx representation to network_diffusion
    net_nd = nd.MultilayerNetwork(layers=net_nx)

    # remove selfloops from the network and actors with no connetcions
    net_nd = nd.mln.functions.remove_selfloop_edges(net_nd)
    actors_to_remove = [
        ac.actor_id
        for ac, nghb in nd.mln.functions.neighbourhood_size(net_nd).items()
        if nghb == 0
    ]
    for l_graph in net_nd.layers.values():
        l_graph.remove_nodes_from(actors_to_remove)

    return net_nd
