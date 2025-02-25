# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A three-layered network with 11 actors made up by Piotr."""

import networkx as nx

from network_diffusion.mln.mlnetwork import MultilayerNetwork


def get_toy_network_piotr() -> MultilayerNetwork:
    """Get threelayered toy network easy to visualise."""
    layer_1 = nx.Graph(
        (
            [1, 4],
            [2, 4],
            [2, 3],
            [3, 4],
            [3, 5],
            [3, 6],
            [4, 5],
            [7, 9],
            [8, 9],
            [8, 10],
        )
    )
    layer_2 = nx.Graph(
        (
            [1, 2],
            [2, 7],
            [2, 11],
            [4, 5],
            [4, 6],
            [5, 11],
            [6, 10],
            [7, 9],
            [8, 9],
            [8, 10],
            [10, 11],
        )
    )
    layer_3 = nx.Graph(
        (
            [1, 4],
            [2, 6],
            [2, 9],
            [3, 4],
            [3, 5],
            [4, 5],
            [5, 6],
            [5, 11],
            [6, 9],
            [7, 9],
            [10, 11],
        )
    )

    return MultilayerNetwork.from_nx_layers(
        [layer_1, layer_2, layer_3], ["l1", "l2", "l3"]
    )


# TODO: add sphinx docs
