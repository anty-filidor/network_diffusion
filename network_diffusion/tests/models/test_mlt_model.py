"""Tests for the network_diffusion.models.mlt_model."""

import random
import unittest

import numpy as np

from network_diffusion.mln.functions import get_toy_network_piotr
from network_diffusion.models import MLTModel
from network_diffusion.seeding.mocking_selector import MockingActorSelector
from network_diffusion.simulator import Simulator
from network_diffusion.utils import fix_random_seed

SPREADING_PARAMETERS = [
    {
        "exp_result": [
            {
                "l1": (("1", 3), ("0", 7)),
                "l2": (("1", 3), ("0", 7)),
                "l3": (("1", 2), ("0", 8)),
            },
            {
                "l1": (("1", 7), ("0", 3)),
                "l2": (("1", 7), ("0", 3)),
                "l3": (("1", 6), ("0", 4)),
            },
            {"l1": (("1", 10),), "l2": (("1", 10),), "l3": (("1", 10),)},
            {"l1": (("1", 10),), "l2": (("1", 10),), "l3": (("1", 10),)},
        ],
        "protocol": "OR",
        "mi_value": 0.3,
    },
    {
        "exp_result": [
            {
                "l1": (("1", 3), ("0", 7)),
                "l2": (("1", 3), ("0", 7)),
                "l3": (("1", 2), ("0", 8)),
            },
            {
                "l1": (("1", 4), ("0", 6)),
                "l2": (("1", 4), ("0", 6)),
                "l3": (("1", 3), ("0", 7)),
            },
            {
                "l1": (("1", 5), ("0", 5)),
                "l2": (("1", 5), ("0", 5)),
                "l3": (("1", 4), ("0", 6)),
            },
            {
                "l1": (("1", 5), ("0", 5)),
                "l2": (("1", 5), ("0", 5)),
                "l3": (("1", 4), ("0", 6)),
            },
        ],
        "protocol": "AND",
        "mi_value": 0.3,
    },
    {
        "exp_result": [
            {
                "l1": (("1", 3), ("0", 7)),
                "l2": (("1", 3), ("0", 7)),
                "l3": (("1", 2), ("0", 8)),
            },
            {
                "l1": (("1", 3), ("0", 7)),
                "l2": (("1", 3), ("0", 7)),
                "l3": (("1", 2), ("0", 8)),
            },
        ],
        "protocol": "AND",
        "mi_value": 0.6,
    },
]


class TestMLTModel(unittest.TestCase):
    """Test class for Multilayer Linear Threshold Model class."""

    def setUp(self) -> None:

        fix_random_seed(42)

        self.network = get_toy_network_piotr()

        # initial seeds for the process choosen arbitrarly and create a ranking
        # that has actors 8 and 1 at the beginning and then the rest of actors
        seeds = [
            self.network.get_actor(8),
            self.network.get_actor(1),
            self.network.get_actor(6),
        ]
        rank = [*seeds, *set(self.network.get_actors()).difference(set(seeds))]
        self.seed_selector = MockingActorSelector(rank)

        # precentages of initially active and inactive actors
        prct_active = len(seeds) / self.network.get_actors_num() * 100
        prct_inactive = 100 - prct_active
        self.budget = (prct_inactive, prct_active)

    def test_parametrised_spreading(self):
        for test_set in SPREADING_PARAMETERS:
            mi_value = test_set["mi_value"]
            protocol = test_set["protocol"]
            exp_result = test_set["exp_result"]
            with self.subTest(
                mi_value=mi_value, protocol=protocol, exp_result=exp_result
            ):
                mltm = MLTModel(
                    protocol=protocol,
                    seed_selector=self.seed_selector,
                    seeding_budget=self.budget,
                    mi_value=mi_value,
                )

                experiment = Simulator(model=mltm, network=self.network.copy())
                logs = experiment.perform_propagation(n_epochs=10, patience=1)

                self.assertEqual(
                    logs._global_stats,
                    exp_result,
                    f"Wrong course of the spreading process, expected "
                    f"{exp_result} found {logs._global_stats} "
                    f"params: ({protocol}, {mi_value})",
                )
