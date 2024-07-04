import os

import networkx as nx
import pytest
import torch
from bidict import bidict

from network_diffusion import utils
from network_diffusion.mln import (
    MultilayerNetwork,
    MultilayerNetworkTorch,
    functions,
    mlnetwork_torch,
)


@pytest.fixture
def set_up():
    utils.fix_random_seed(42)


def network_florentine():
    return MultilayerNetwork.from_mpx(
        f"{utils.get_absolute_path()}/tests/data/florentine.mpx"
    )


@pytest.mark.parametrize(
    "net_initial,exp_ac_map,exp_added_nodes",
    [
        (
            functions.get_toy_network_piotr(),
            bidict(
                {
                    1: 0,
                    4: 1,
                    2: 2,
                    3: 3,
                    5: 4,
                    6: 5,
                    7: 6,
                    9: 7,
                    8: 8,
                    10: 9,
                    11: 10,
                }
            ),
            {"l1": {11}, "l2": {3}, "l3": {8}},
        ),
        (
            functions.get_toy_network_cim(),
            bidict(
                {
                    1: 0,
                    2: 1,
                    3: 2,
                    4: 3,
                    6: 4,
                    7: 5,
                    5: 6,
                    11: 7,
                    8: 8,
                    9: 9,
                    10: 10,
                    12: 11,
                }
            ),
            None,
        ),
        (
            network_florentine(),
            bidict(
                {
                    "Ridolfi": 0,
                    "Tornabuoni": 1,
                    "Strozzi": 2,
                    "Peruzzi": 3,
                    "Pazzi": 4,
                    "Salviati": 5,
                    "Medici": 6,
                    "Guadagni": 7,
                    "Lamberteschi": 8,
                    "Castellani": 9,
                    "Bischeri": 10,
                    "Barbadori": 11,
                    "Albizzi": 12,
                    "Ginori": 13,
                    "Acciaiuoli": 14,
                }
            ),
            {
                "marriage": set(),
                "business": {"Strozzi", "Acciaiuoli", "Ridolfi", "Albizzi"},
            },
        ),
    ],
)
def test__prepare_mln_for_conversion(
    net_initial, exp_ac_map, exp_added_nodes, set_up
):
    obt_net, obt_actors, obt_added_nodes = (
        mlnetwork_torch._prepare_mln_for_conversion(net_initial)
    )
    assert obt_net.is_multiplex()
    assert obt_added_nodes == exp_added_nodes
    assert obt_actors == exp_ac_map


# @pytest.mark.parametrize(
#         "tensor_raw,tensor_coalesced",
#         [
#             (),
#         ]
# )
# def test__coalesce_raw_tensor(tensor_raw, tensor_coalesced):
#     assert torch.all(tensor_raw._indices() == tensor_coalesced._indices())
#     assert torch.all(tensor_raw._values() == tensor_coalesced._values())
#     assert tensor_raw.size() == tensor_coalesced.size()
#     assert tensor_raw._nnz() == tensor_coalesced._nnz()
#     assert tensor_raw.layout == tensor_coalesced.layout


# @pytest.mark.parametrize(
#     "raw_tensor,exp_tensor",
#     [
#         (

# torch.sparse_coo_tensor(
#     indices=torch.Tensor(
#         [[0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9],
#         [1, 0, 2, 3, 4, 1, 3, 1, 2, 4, 5, 1, 3, 3, 7, 6, 8, 7, 9, 8]]
#     ),
#     values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     size=(11, 11),
#     is_coalesced=False,
# )

# torch.sparse_coo_tensor(
#     indices=torch.Tensor(
#         [[ 0, 1, 1, 2, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7,8, 8, 9, 9, 9, 10, 10, 10],
#         [2,  4,  5,  0,  6, 10,  1, 10,  1,  9,  2,  7,  6,  8, 7,  9,  5,  8, 10,  2,  4,  9]]
#     ),
#     values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     size=(11, 11),
#     is_coalesced=False,
# )

# torch.sparse_coo_tensor(
#     indices=torch.Tensor(
#         [[ 0,  1,  1,  1,  2,  2,  3,  3,  4,  4,  4,  4,  5,  5, 5,  6,  7,  7,  7,  9, 10, 10],
#         [ 1,  0,  3,  4,  5,  7,  1,  4,  1,  3,  5, 10,  2,  4, 7,  7,  2,  5,  6, 10,  4,  9]]
#     ),
#     values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     size=(11, 11),
#     is_coalesced=False,
# )

#         )

#     ]

# def test__mln_to_sparse():
#     net = functions.get_toy_network_piotr()
#     net_converted, ac_map, nodes_added = mlnetwork_torch._prepare_mln_for_conversion(net)
#     mlnetwork_torch._mln_to_sparse(net_converted, list(ac_map.values()))
#     assert True


@pytest.mark.parametrize(
    "layers_order,actors_map,nodes_added,exp_output",
    [
        (
            ["l1", "l2"],
            bidict({1: 0, 2: 1, 3: 2, 4: 3}),
            {"l1": {}, "l2": {1, 2}},
            torch.Tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        ),
        (
            ["l_1"],
            bidict({0: 0, 1: 1, 2: 2}),
            {"l_1": {}},
            torch.Tensor([[0.0, 0.0, 0.0]]),
        ),
        (
            ["la_1", "la_2", "la_3"],
            bidict({0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4}),
            {"la_1": {0, 1, 2}, "la_2": {}, "la_3": {5}},
            torch.Tensor(
                [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ),
    ],
)
def test__create_nodes_mask(
    layers_order, actors_map, nodes_added, exp_output, set_up
):
    obtained_mask = mlnetwork_torch._create_nodes_mask(
        layers_order, actors_map, nodes_added, True
    )
    assert torch.equal(exp_output, obtained_mask)


# class TestMultilayerNetworkTorch:

#     @pytest.mark.parametrize(
#         "network,exp_output",
#         [
#             (
#                 functions.get_toy_network_piotr(),
#                 # bidict({1:0, 2:1, 3:2, 4:3}),
#                 # {"l1": {}, "l2": {1, 2}},
#                 torch.Tensor([[0., 0., 0., 0.], [1., 1., 0., 0.]])
#             ),
#         ]
#     )
#     def test_from_mln(self, network, exp_output):
#         net_tensor = MultilayerNetworkTorch.from_mln(network)
#         assert True

#     def test___repr__(self):
#         assert True
