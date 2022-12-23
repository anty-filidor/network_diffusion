"""Randomised seed selector for DSAA algorithm."""
from random import shuffle
from typing import List, Tuple

from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.multilayer_network import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector


class RandomSeedSelector(BaseSeedSelector):
    """Randomised seed selector for DSAA algorithm."""

    # pylint: disable=R0914

    def __call__(
        self, compartmental: CompartmentalGraph, network: MultilayerNetwork
    ) -> List[NetworkUpdateBuffer]:
        """Populate seeds ranking list by random choice."""
        model_hyperparams = compartmental.get_compartments()
        states_seeds = compartmental.get_seeding_budget_for_network(network)

        ranking_list: List[NetworkUpdateBuffer] = []

        # set initial states in each layer of network
        for layer_name, layer_graph in network.layers.items():

            states: Tuple[str, ...] = model_hyperparams[
                layer_name
            ]  # type: ignore
            budget_in_layer = states_seeds[layer_name]
            nodes_in_layer_num = len(layer_graph.nodes())

            assert nodes_in_layer_num == sum(budget_in_layer)

            # shuffle nodes
            nodes = [*layer_graph.nodes()]
            shuffle(nodes)

            # set ranges - idk what is it
            _rngs = [
                sum(budget_in_layer[:x]) for x in range(len(budget_in_layer))
            ] + [nodes_in_layer_num]
            ranges: List[Tuple[int, int]] = list(zip(_rngs[:-1], _rngs[1:]))

            # append states to nodes
            for i, _ in enumerate(ranges):
                pair = ranges[i]
                state = states[i]
                for index in range(pair[0], pair[1]):
                    ranking_list.append(
                        NetworkUpdateBuffer(
                            node_name=nodes[index],
                            layer_name=layer_name,
                            new_state=state,
                        )
                    )

        return ranking_list
