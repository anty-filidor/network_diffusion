import warnings
from multiprocessing.managers import SharedMemoryManager

import pytest

from network_diffusion.mln.mds.greedy_search import get_mds_greedy
from network_diffusion.mln.mds.utils import (
    ShareableListManager,
    is_dominating_set,
)
from network_diffusion.nets import get_toy_network_piotr


def test_is_dominating_set():
    net = get_toy_network_piotr()
    mds = get_mds_greedy(net=net)
    assert is_dominating_set(mds, net)
    mds.pop()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        assert not is_dominating_set(mds, net)


@pytest.fixture
def smm():
    """Fixture to start and stop SharedMemoryManager."""
    with SharedMemoryManager() as manager:
        yield manager


class TestShareableListManager:

    def test_initialisation(self, smm):
        length = 5
        slm = ShareableListManager(smm, length)
        assert len(slm.sl) == length
        assert all(
            item is ShareableListManager.placeholder_token for item in slm.sl
        )

    def test_update_sl_partial(self, smm):
        slm = ShareableListManager(smm, 4)
        slm.update_sl([1, 2])
        assert slm.sl[0] == 1
        assert slm.sl[1] == 2
        assert slm.sl[2] is ShareableListManager.placeholder_token
        assert slm.sl[3] is ShareableListManager.placeholder_token

    def test_update_sl_full(self, smm):
        slm = ShareableListManager(smm, 3)
        slm.update_sl(["a", "b", "c"])
        assert [ssl for ssl in slm.sl] == ["a", "b", "c"]

    def test_update_sl_longer_input(self, smm):
        slm = ShareableListManager(smm, 2)
        slm.update_sl(["x", "y", "z"])
        assert [ssl for ssl in slm.sl] == ["x", "y"]

    def test_get_as_pruned_set(self, smm):
        slm = ShareableListManager(smm, 5)
        slm.update_sl([1, None, 2])
        result = slm.get_as_pruned_set()
        assert result == {1, 2}
        assert ShareableListManager.placeholder_token not in result
