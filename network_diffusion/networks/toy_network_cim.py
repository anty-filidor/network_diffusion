"""A toy network used with by the authors of CIM method."""

import networkx as nx

from network_diffusion.mln.mlnetwork import MultilayerNetwork


def get_toy_network_cim() -> MultilayerNetwork:
    """
    Get a toy network that was used by the authors of CIM method.

    The paper is here: https://doi.org/10.1007/s10489-021-02656-0
    """
    actors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    layer_1 = nx.Graph(
        (
            [1, 2],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
            [3, 6],
            [3, 7],
            [4, 5],
            [4, 11],
            [5, 7],
            [5, 11],
            [6, 7],
            [7, 8],
            [7, 9],
            [7, 10],
            [8, 9],
            [8, 10],
            [9, 10],
            [9, 12],
            [10, 12],
            [11, 12],
        )
    )
    layer_1.add_nodes_from(actors)

    layer_2 = nx.Graph(
        (
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 3],
            [2, 4],
            [2, 5],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [4, 11],
            [5, 11],
            [6, 7],
            [6, 8],
            [6, 9],
            [6, 10],
            [6, 12],
            [7, 8],
            [7, 9],
            [7, 10],
            [7, 12],
            [8, 9],
            [8, 10],
            [8, 12],
            [9, 10],
            [9, 12],
            [10, 12],
        )
    )
    layer_2.add_nodes_from(actors)

    layer_3 = nx.Graph(
        (
            [2, 4],
            [3, 6],
            [3, 7],
            [3, 8],
            [3, 9],
            [3, 10],
            [3, 11],
            [3, 12],
            [6, 7],
            [6, 8],
            [6, 9],
            [6, 10],
            [6, 11],
            [6, 12],
            [7, 8],
            [7, 9],
            [7, 10],
            [7, 11],
            [7, 12],
            [8, 9],
            [8, 10],
            [8, 11],
            [8, 12],
            [9, 10],
            [9, 11],
            [9, 12],
            [10, 11],
            [10, 12],
            [11, 12],
        )
    )
    layer_3.add_nodes_from(actors)

    return MultilayerNetwork.from_nx_layers(
        [layer_1, layer_2, layer_3], ["l1", "l2", "l3"]
    )


# TODO: add sphinx docs
