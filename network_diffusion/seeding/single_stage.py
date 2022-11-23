from random import shuffle
from typing import Any, Dict, List, NamedTuple, Tuple

from network_diffusion.models.base import NetworkUpdateBuffer
from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.multilayer_network import MultilayerNetwork

# network
# obtain seeding budget from model
# use some seeds
# evaluate network
# obtain list of nodes to update
# if list is too small then apply heuristic to use some seeding budget
# update the network


class RandomSeedSelection:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        model: CompartmentalGraph,
        network: MultilayerNetwork,
    ) -> None:
        """
        Prepare network to be used during simulation.

        It changes status of certain nodes as passed in states_seeds. After
        execution of this method experiment is reeady to be done.

        :param states_seeds: dictionary with numbers of nodes to be initialised
            in each layer of network. For example following argument:
            {'illness': (75, 2, 0), 'awareness': (60, 17), 'vaccination':
            (70, 7) } is correct with this model of propagation: [ illness :
            ('S', 'I', 'R'), awareness : ('UA', 'A'), vaccination : ('UV', 'V')
            ]. Note, that values in dictionary says how many nodes should have
            state <x> at the beginning of the experiment.
        """
        # pylint: disable=R0914

        model_hyperparams = model.get_model_hyperparams()
        states_seeds = model.get_seeding_budget_for_network(network)

        nodes_to_update: List[NetworkUpdateBuffer] = []

        # set initial states in each layer of network
        for layer_name, layer_graph in network.layers.items():
            states = model_hyperparams[layer_name]
            vals = states_seeds[layer_name]
            nodes_number = len(layer_graph.nodes())

            # shuffle nodes
            nodes = [*layer_graph.nodes()]
            shuffle(nodes)

            # set ranges
            _rngs = [sum(vals[:x]) for x in range(len(vals))] + [nodes_number]
            ranges: List[Tuple[int, int]] = list(zip(_rngs[:-1], _rngs[1:]))

            # append states to nodes
            for i, _ in enumerate(ranges):
                pair = ranges[i]
                state = states[i]
                for index in range(pair[0], pair[1]):
                    nodes_to_update.append(
                        NetworkUpdateBuffer(
                            node_name=nodes[index],
                            layer_name=layer_name,
                            new_state=state,
                        )
                    )
                    # layer.nodes[nodes[index]]["status"] = state
        return nodes_to_update
