import unittest

from network_diffusion.mln.functions import get_toy_network_cim
from network_diffusion.mln.kppshell import (
    compute_seed_quotas,
    get_toy_network_kppshell,
    kppshell_decomposition,
    kppshell_seed_ranking,
    kppshell_seed_selection,
)

KPP_SEED_SETS = [
    {"seed_num": 1, "exp_seed_set": {1}},
    {"seed_num": 2, "exp_seed_set": {1, 6}},
    {"seed_num": 3, "exp_seed_set": {1, 6, 7}},
    {"seed_num": 4, "exp_seed_set": {1, 3, 6, 7}},
    {"seed_num": 5, "exp_seed_set": {1, 3, 6, 7, 9}},
    {"seed_num": 6, "exp_seed_set": {1, 3, 4, 6, 7, 9}},
    {"seed_num": 7, "exp_seed_set": {1, 3, 4, 5, 6, 7, 9}},
    {"seed_num": 8, "exp_seed_set": {1, 3, 4, 5, 6, 7, 9, 10}},
    {"seed_num": 9, "exp_seed_set": {1, 2, 3, 4, 5, 6, 7, 9, 10}},
    {"seed_num": 10, "exp_seed_set": {1, 2, 3, 4, 5, 6, 7, 9, 10, 12}},
    {"seed_num": 11, "exp_seed_set": {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12}},
    {"seed_num": 12, "exp_seed_set": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
]

KPP_QUOTAS_1 = [
    {"quota": 1, "exp_division": [1, 0]},
    {"quota": 2, "exp_division": [1, 1]},
    {"quota": 3, "exp_division": [2, 1]},
    {"quota": 4, "exp_division": [3, 1]},
    {"quota": 5, "exp_division": [3, 2]},
    {"quota": 6, "exp_division": [4, 2]},
    {"quota": 7, "exp_division": [5, 2]},
    {"quota": 8, "exp_division": [5, 3]},
    {"quota": 9, "exp_division": [6, 3]},
    {"quota": 10, "exp_division": [6, 4]},
    {"quota": 11, "exp_division": [7, 4]},
    {"quota": 12, "exp_division": [7, 5]},
]

KPP_QUOTAS_2 = [
    {"quota": 1, "exp_division": [0, 1, 0, 0]},
    {"quota": 2, "exp_division": [1, 1, 0, 0]},
    {"quota": 3, "exp_division": [1, 2, 0, 0]},
    {"quota": 4, "exp_division": [1, 2, 1, 0]},
    {"quota": 5, "exp_division": [1, 3, 1, 0]},
    {"quota": 6, "exp_division": [1, 4, 1, 0]},
    {"quota": 7, "exp_division": [1, 4, 1, 1]},
    {"quota": 8, "exp_division": [1, 5, 1, 1]},
    {"quota": 9, "exp_division": [1, 6, 1, 1]},
    {"quota": 10, "exp_division": [1, 7, 1, 1]},
    {"quota": 11, "exp_division": [1, 8, 1, 1]},
    {"quota": 12, "exp_division": [2, 8, 1, 1]},
]


class TestKPPShell(unittest.TestCase):

    def setUp(self):
        self.network_1 = get_toy_network_kppshell()["l1"]
        self.communities_1 = [{1, 2, 3, 4, 5, 7, 11}, {6, 8, 9, 10, 12}]
        self.network_2 = get_toy_network_cim()["l3"]
        self.communities_2 = [{2, 4}, {3, 6, 7, 8, 9, 10, 11, 12}, {1}, {5}]

    def test_kppshell_decomposition(self):
        shells = kppshell_decomposition(self.network_1)
        expected_shells = [
            [
                {"node_id": 11, "shell": 1, "reward_points": 0},
                {"node_id": 2, "shell": 2, "reward_points": 0},
                {"node_id": 5, "shell": 2, "reward_points": 1},
                {"node_id": 1, "shell": 3, "reward_points": 2},
                {"node_id": 3, "shell": 3, "reward_points": 0},
                {"node_id": 4, "shell": 3, "reward_points": 0},
                {"node_id": 7, "shell": 3, "reward_points": 2},
            ],
            [
                {"node_id": 8, "shell": 1, "reward_points": 0},
                {"node_id": 6, "shell": 2, "reward_points": 1},
                {"node_id": 9, "shell": 2, "reward_points": 0},
                {"node_id": 10, "shell": 2, "reward_points": 0},
                {"node_id": 12, "shell": 2, "reward_points": 0},
            ],
        ]
        self.assertEqual(shells, expected_shells)

    def test_kppshell_seed_ranking(self):
        ranking = kppshell_seed_ranking(self.network_1)
        expected_ranking = [1, 6, 7, 3, 9, 4, 5, 10, 2, 12, 11, 8]
        self.assertEqual(ranking, expected_ranking)

    def test_kppshell_seed_selection(self):
        for test_case in KPP_SEED_SETS:
            seed_num = test_case["seed_num"]
            exp_seed_set = test_case["exp_seed_set"]
            seed_set = kppshell_seed_selection(self.network_1, seed_num)
            self.assertEqual(seed_set, exp_seed_set)

    def test_compute_seed_quota_network_1(self):
        for test_case in KPP_QUOTAS_1:
            quota = test_case["quota"]
            exp_division = test_case["exp_division"]
            division = compute_seed_quotas(
                self.network_1, self.communities_1, quota
            )
            self.assertEqual(division, exp_division)

    def test_compute_seed_quota_network_2(self):
        for test_case in KPP_QUOTAS_2:
            quota = test_case["quota"]
            exp_division = test_case["exp_division"]
            division = compute_seed_quotas(
                self.network_2, self.communities_2, quota
            )
            self.assertEqual(division, exp_division)


if __name__ == "__main__":
    unittest.main(verbosity=2)
