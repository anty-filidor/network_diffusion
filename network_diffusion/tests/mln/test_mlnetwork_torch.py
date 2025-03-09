from unittest.mock import MagicMock

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
from network_diffusion.nets import (
    get_l2_course_net,
    get_toy_network_cim,
    get_toy_network_piotr,
)


@pytest.fixture
def set_up():
    utils.fix_random_seed(42)


class MockTensor(MagicMock):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def to(self, device):
        self.device = device
        return self


def network_florentine():
    return MultilayerNetwork.from_mpx(
        f"{utils._get_absolute_path()}/tests/data/florentine.mpx"
    )


def network_t1():
    return MultilayerNetwork(
        {
            "l1": nx.Graph([(0, 4), (1, 2), (1, 3), (1, 4), (2, 3)]),
            "l2": nx.Graph([(0, 2), (1, 4), (2, 3)]),
        }
    )


def network_t2():
    net_base = network_t1()
    net_base["l1"].add_node(5)
    net_base["l2"].add_edge(0, 5)
    return MultilayerNetwork({"l1": net_base["l1"], "l2": net_base["l2"]})


def network_t2_hard():
    net_base = network_t1()
    net_base["l2"].add_edge(0, 5)
    return MultilayerNetwork({"l1": net_base["l1"], "l2": net_base["l2"]})


def compare_sparse_tensors(t1, t2):
    assert torch.all(t1._indices() == t2._indices())
    assert torch.all(t1._values() == t2._values())
    assert t1.size() == t2.size()
    assert t1._nnz() == t2._nnz()
    assert t1.layout == t2.layout


@pytest.mark.parametrize(
    "net_in,exp_ac_map,exp_ad_nodes",
    [
        (
            get_toy_network_piotr(),
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
            get_toy_network_cim(),
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
            get_l2_course_net(False, False, False).snaps[2],
            bidict(
                {
                    1: 0,
                    8: 1,
                    14: 2,
                    24: 3,
                    2: 4,
                    4: 5,
                    6: 6,
                    19: 7,
                    28: 8,
                    29: 9,
                    37: 10,
                    39: 11,
                    3: 12,
                    13: 13,
                    25: 14,
                    26: 15,
                    27: 16,
                    32: 17,
                    33: 18,
                    35: 19,
                    11: 20,
                    17: 21,
                    30: 22,
                    31: 23,
                    5: 24,
                    12: 25,
                    22: 26,
                    7: 27,
                    15: 28,
                    21: 29,
                    38: 30,
                    9: 31,
                    10: 32,
                    23: 33,
                    34: 34,
                    41: 35,
                    20: 36,
                    18: 37,
                    36: 38,
                    40: 39,
                    16: 40,
                }
            ),
            None,
        ),
    ],
)
def test__prepare_mln_for_conversion(net_in, exp_ac_map, exp_ad_nodes, set_up):
    obt_net, obt_actors, obt_added_nodes = (
        mlnetwork_torch._prepare_mln_for_conversion(net_in)
    )
    assert obt_net.is_multiplex()
    assert obt_added_nodes == exp_ad_nodes
    assert obt_actors == exp_ac_map


@pytest.mark.parametrize(
    "net_in,ac_order,net_exp,l_exp",
    [
        (
            network_t1(),
            [0, 1, 2, 3, 4],
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 0, 1, 2, 2, 3, 4],
                        [4, 2, 3, 4, 1, 3, 1, 2, 0, 1, 2, 4, 0, 3, 2, 1],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
        ),
        (
            network_t2(),
            [0, 1, 2, 3, 4, 5],
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 2, 2, 3, 4, 5],
                        [4, 2, 3, 4, 1, 3, 1, 2, 0, 1, 2, 5, 4, 0, 3, 2, 1, 0],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
        ),
        (
            network_t2(),
            [5, 3, 1, 0, 2, 4],
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 2, 2, 3, 4, 4, 5, 5, 0, 1, 2, 3, 3, 4, 4, 5],
                        [2, 4, 1, 4, 5, 5, 1, 2, 2, 3, 3, 4, 5, 0, 4, 1, 3, 2],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
        ),
    ],
)
def test__mln_to_sparse(net_in, ac_order, net_exp, l_exp):
    net_obtained, l_names = mlnetwork_torch._mln_to_sparse(net_in, ac_order)
    assert l_exp == l_names
    compare_sparse_tensors(net_exp, net_obtained)


@pytest.mark.parametrize(
    "l_order,ac_map,nodes_added,exp_output",
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
def test__create_nodes_mask(l_order, ac_map, nodes_added, exp_output, set_up):
    obtained_mask = mlnetwork_torch._create_nodes_mask(
        l_order, ac_map, nodes_added, True
    )
    assert torch.equal(exp_output, obtained_mask)


@pytest.mark.parametrize(
    "net_raw,exp_adt,exp_l_order,exp_ac_map,exp_n_mask",
    [
        (
            network_t1(),
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 0, 1, 2, 3, 3, 4],
                        [1, 0, 2, 1, 3, 4, 2, 4, 2, 3, 3, 2, 1, 0, 4, 3],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
            bidict({0: 0, 4: 1, 1: 2, 2: 3, 3: 4}),
            torch.Tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
            ),
        ),
        (
            network_t2(),
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 0, 0, 1, 2, 3, 3, 4, 5],
                        [1, 0, 2, 1, 3, 4, 2, 4, 2, 3, 3, 5, 2, 1, 0, 4, 3, 0],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
            bidict({0: 0, 4: 1, 1: 2, 2: 3, 3: 4, 5: 5}),
            torch.Tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
        (
            network_t2_hard(),
            torch.sparse_coo_tensor(
                indices=torch.Tensor(
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 0, 0, 1, 2, 3, 3, 4, 5],
                        [1, 0, 2, 1, 3, 4, 2, 4, 2, 3, 3, 5, 2, 1, 0, 4, 3, 0],
                    ]
                ),
                values=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ),
            ["l1", "l2"],
            bidict({0: 0, 4: 1, 1: 2, 2: 3, 3: 4, 5: 5}),
            torch.Tensor(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
        ),
    ],
)
def test_MultilayerNetworkTorch_from_mln(
    net_raw, exp_adt, exp_l_order, exp_ac_map, exp_n_mask
):
    net_t = MultilayerNetworkTorch.from_mln(net_raw, "cpu")
    compare_sparse_tensors(net_t.adjacency_tensor, exp_adt)
    assert net_t.layers_order == exp_l_order
    assert net_t.actors_map == exp_ac_map
    assert torch.all(net_t.nodes_mask == exp_n_mask)
    print(net_t.device)


def test_MultilayerNetworkTorch_device_get_exception():
    net_t = MultilayerNetworkTorch.from_mln(get_toy_network_piotr())
    net_t.adjacency_tensor = MockTensor("cuda:0")
    with pytest.raises(ValueError):
        net_t.device


def test_MultilayerNetworkTorch_device_get():
    net_t = MultilayerNetworkTorch.from_mln(get_toy_network_piotr())
    net_t.adjacency_tensor = MockTensor()
    net_t.nodes_mask = MockTensor()
    net_t.device = "cuda:0"
    assert net_t.device == "cuda:0"


def test_MultilayerNetworkTorch_copy():
    net_0 = MultilayerNetworkTorch.from_mln(network_florentine())
    net_1 = net_0.copy()

    for attr in [
        "adjacency_tensor",
        "layers_order",
        "actors_map",
        "nodes_mask",
        "device",
    ]:
        attr_0 = net_0.__getattribute__(attr)
        attr_1 = net_1.__getattribute__(attr)
        assert id(attr_0) != id(attr_1)

    assert net_0.layers_order == net_1.layers_order
    assert net_0.actors_map == net_1.actors_map
    assert net_0.device == net_1.device
    assert torch.all(net_0.nodes_mask == net_1.nodes_mask)
    compare_sparse_tensors(net_0.adjacency_tensor, net_1.adjacency_tensor)
