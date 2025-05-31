import warnings

import pytest

from network_diffusion.mln.mds.greedy_search import get_mds_greedy
from network_diffusion.mln.mds.utils import is_dominating_set
from network_diffusion.nets import get_l2_course_net, get_toy_network_piotr
from network_diffusion.utils import set_rng_seed


def test_is_dominating_set():
    net = get_toy_network_piotr()
    mds = get_mds_greedy(net=net)
    assert is_dominating_set(mds, net)
    mds.pop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        assert not is_dominating_set(mds, net)


@pytest.mark.parametrize(
    "network, mds",
    [
        (
            get_toy_network_piotr(),
            {2, 3, 4, 8, 9, 11},
        ),
        (
            get_l2_course_net(False, False, False)[0],
            {1, 2, 3, 9, 11, 22, 24, 26, 31, 40},
        ),
        (
            get_l2_course_net(False, False, False)[1],
            {3, 5, 37, 39, 11, 15, 18, 24},
        ),
    ],
)
def test_get_mds_greedy(network, mds):
    set_rng_seed(42)
    driver_actors = get_mds_greedy(net=network)
    assert {a.actor_id for a in driver_actors} == mds
