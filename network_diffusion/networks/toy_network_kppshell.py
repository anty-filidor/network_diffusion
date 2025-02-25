# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""A toy network that was used by the authors of K++ Shell method."""

import networkx as nx

from network_diffusion.mln.mlnetwork import MultilayerNetwork


def get_toy_network_kppshell() -> MultilayerNetwork:
    """
    Get a toy network that was used by the authors of K++ Shell method.

    The paper is here: https://doi.org/10.1016/j.comnet.2023.109916
    """
    actors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    layer_1 = nx.Graph(
        (
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 7],
            [2, 7],
            [3, 4],
            [3, 7],
            [4, 7],
            [5, 7],
            [5, 6],
            [5, 11],
            [6, 8],
            [6, 9],
            [6, 10],
            [9, 12],
            [10, 12],
        )
    )
    layer_1.add_nodes_from(actors)

    return MultilayerNetwork.from_nx_layers([layer_1], ["l1"])


# TODO: add sphinx docs
