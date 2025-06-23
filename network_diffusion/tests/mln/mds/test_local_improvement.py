import pytest

from network_diffusion.mln.mds.local_improvement import (
    LocalImprovement,
    get_mds_locimpr,
)
from network_diffusion.mln.mds.utils import is_dominating_set
from network_diffusion.nets import get_l2_course_net, get_toy_network_piotr
from network_diffusion.utils import set_rng_seed


@pytest.mark.parametrize(
    "network, mds, timeout",
    [
        (
            get_toy_network_piotr(),
            {2, 3, 4, 8, 9, 11},
            0.001,
        ),
        (
            get_l2_course_net(False, False, False)[0],
            {1, 2, 3, 40, 9, 11, 22, 24, 26, 31},
            0.001,
        ),
        (
            get_l2_course_net(False, False, False)[1],
            {3, 5, 37, 39, 11, 15, 18, 24},
            0.001,
        ),
        (
            get_toy_network_piotr(),
            {2, 3, 4, 9, 10},
            None,
        ),
        (
            get_l2_course_net(False, False, False)[2],
            {2, 3, 7, 22, 24, 27, 28, 30},
            None,
        ),
    ],
)
def test_get_mds_locimpr(network, mds, timeout):
    set_rng_seed(42)
    driver_actors = get_mds_locimpr(net=network, timeout=timeout, debug=False)
    obt_mds = {a.actor_id for a in driver_actors}
    # a workaround for many combinations of the same size MDS under a given RNG
    assert len(obt_mds) <= len(mds)
    assert is_dominating_set(driver_actors, network)


@pytest.fixture
def li():
    return LocalImprovement(net=get_toy_network_piotr(), timeout=60)


class TestLocalImprovement:

    @pytest.mark.parametrize(
        "dominating_set, exp_map",
        [
            (
                {2, 3, 4, 9, 10},
                {
                    "l1": {
                        1: {4},
                        4: {2, 3, 4},
                        2: {2, 3, 4},
                        3: {2, 3, 4},
                        5: {3, 4},
                        6: {3},
                        7: {9},
                        9: {9},
                        8: {9, 10},
                        10: {10},
                        11: set(),
                    },
                    "l2": {
                        1: {2},
                        4: {4},
                        2: {2},
                        3: set(),
                        5: {4},
                        6: {10, 4},
                        7: {9, 2},
                        9: {9},
                        8: {9, 10},
                        10: {10},
                        11: {2, 10},
                    },
                    "l3": {
                        1: {4},
                        4: {3, 4},
                        2: {9, 2},
                        3: {3, 4},
                        5: {3, 4},
                        6: {9, 2},
                        7: {9},
                        9: {9, 2},
                        8: set(),
                        10: {10},
                        11: {10},
                    },
                },
            ),
            (
                {"nonexisting_actor"},
                {
                    "l1": {
                        1: set(),
                        4: set(),
                        2: set(),
                        3: set(),
                        5: set(),
                        6: set(),
                        7: set(),
                        9: set(),
                        8: set(),
                        10: set(),
                        11: set(),
                    },
                    "l2": {
                        1: set(),
                        4: set(),
                        2: set(),
                        3: set(),
                        5: set(),
                        6: set(),
                        7: set(),
                        9: set(),
                        8: set(),
                        10: set(),
                        11: set(),
                    },
                    "l3": {
                        1: set(),
                        4: set(),
                        2: set(),
                        3: set(),
                        5: set(),
                        6: set(),
                        7: set(),
                        9: set(),
                        8: set(),
                        10: set(),
                        11: set(),
                    },
                },
            ),
            (
                {1},
                {
                    "l1": {
                        1: {1},
                        4: {1},
                        2: set(),
                        3: set(),
                        5: set(),
                        6: set(),
                        7: set(),
                        9: set(),
                        8: set(),
                        10: set(),
                        11: set(),
                    },
                    "l2": {
                        1: {1},
                        4: set(),
                        2: {1},
                        3: set(),
                        5: set(),
                        6: set(),
                        7: set(),
                        9: set(),
                        8: set(),
                        10: set(),
                        11: set(),
                    },
                    "l3": {
                        1: {1},
                        4: {1},
                        2: set(),
                        3: set(),
                        5: set(),
                        6: set(),
                        7: set(),
                        9: set(),
                        8: set(),
                        10: set(),
                        11: set(),
                    },
                },
            ),
        ],
    )
    def test_compute_domination(self, li, dominating_set, exp_map):
        dom = li._compute_domination(dominating_set)
        assert dom == exp_map

    @pytest.mark.parametrize(
        "u, dominating_set, exp_res",
        [
            (2, {2, 3, 4, 9, 10}, [1]),
            (3, {2, 3, 4, 9, 10}, [6]),
            (11, {2, 3, 4, 9, 10, 11}, [1, 5, 6, 7]),
        ],
    )
    def test_find_replacement_candidates(self, li, u, dominating_set, exp_res):
        domination = li._compute_domination(dominating_set)
        candidates = li._find_replacement_candidates(
            u, dominating_set, domination
        )
        assert isinstance(candidates, list)
        assert set(candidates) == set(exp_res)

    @pytest.mark.parametrize(
        "dominating_set, exp_res",
        [
            ({2, 3, 4, 9, 10}, True),
            ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, True),
            ({1}, False),
            ({2, 3, 4}, False),
        ],
    )
    def test_is_feasible(self, li, dominating_set, exp_res):
        assert li._is_feasible(dominating_set) is exp_res

    @pytest.mark.parametrize(
        "dominating_set, exp_res",
        [
            ({2, 3, 4, 9, 10}, {2, 3, 4, 9, 10}),
            ({2, 3, 4, 8, 9, 10, 11}, {2, 3, 4, 9, 10}),
        ],
    )
    def test_remove_redundant_vertices(self, li, dominating_set, exp_res):
        reduced = li._remove_redundant_vertices(dominating_set)
        assert reduced == exp_res
        assert li._is_feasible(reduced)
        assert len(reduced) <= len(dominating_set)
