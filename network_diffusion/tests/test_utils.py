import unittest

import numpy as np

from network_diffusion import utils


class TestUtils(unittest.TestCase):
    """Test class for utils script."""

    def test_set_rng_seed(self):
        utils.set_rng_seed(12345)
        exp_arr = [99, 30, 2, 37, 42, 35, 30, 2, 60, 15, 92, 81, 74, 12, 78]
        assert np.array_equal(np.random.randint(1, 100, 15), exp_arr)


if __name__ == "__main__":
    unittest.main()
