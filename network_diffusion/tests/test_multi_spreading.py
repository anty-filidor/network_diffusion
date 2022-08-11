#!/usr/bin/env python3

import os
import random
import shutil
import string
import unittest
from typing import Any, Dict, Tuple
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np
import pandas as pd

from network_diffusion import (
    MultilayerNetwork,
    MultiSpreading,
    PropagationModel,
)
from network_diffusion.multi_spreading import ExperimentLogger


def prepare_data() -> Tuple[
    PropagationModel, Dict[str, Tuple[str, ...]], MultilayerNetwork
]:
    """Sets up data needed to perform tests."""
    # init propagation model and set transitions with probabilities
    model = PropagationModel()
    phenomenas = {
        "ill": ("S", "I", "R"),
        "aware": ("UA", "A"),
        "vacc": ("UV", "V"),
    }
    for l, p in phenomenas.items():
        model.add(l, p)
    model.compile(background_weight=0)

    # init multilayer network from nx predefined network
    network = MultilayerNetwork()
    network.load_layer_nx(nx.les_miserables_graph(), [*phenomenas.keys()])

    model.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.UA"), 0.9)
    model.set_transition_fast("ill.S", "ill.I", ("vacc.V", "aware.A"), 0.05)
    model.set_transition_fast("ill.S", "ill.I", ("vacc.UV", "aware.A"), 0.2)
    model.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.UA"), 0.1)
    model.set_transition_fast("ill.I", "ill.R", ("vacc.V", "aware.A"), 0.7)
    model.set_transition_fast("ill.I", "ill.R", ("vacc.UV", "aware.A"), 0.3)
    model.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.S"), 0.03)
    model.set_transition_fast("vacc.UV", "vacc.V", ("aware.A", "ill.I"), 0.01)
    model.set_transition_fast(
        "aware.UA", "aware.A", ("vacc.UV", "ill.S"), 0.05
    )
    model.set_transition_fast("aware.UA", "aware.A", ("vacc.V", "ill.S"), 1)
    model.set_transition_fast("aware.UA", "aware.A", ("vacc.UV", "ill.I"), 0.2)

    return model, phenomenas, network


def prepare_logs() -> Tuple[Dict[str, Tuple[Any, ...]]]:
    """Prepares logs for tests of ExperimentLogger class."""
    return (
        {
            "ill": (("S", 47), ("I", 30)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 9), ("V", 68)),
        },
        {
            "ill": (("S", 38), ("I", 39)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 9), ("V", 68)),
        },
        {
            "ill": (("I", 42), ("S", 35)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 9), ("V", 68)),
        },
        {
            "ill": (("I", 49), ("S", 28)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 6), ("V", 71)),
        },
        {
            "ill": (("I", 56), ("S", 21)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 6), ("V", 71)),
        },
        {
            "ill": (("I", 58), ("S", 19)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 5), ("V", 72)),
        },
        {
            "ill": (("I", 60), ("S", 17)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 5), ("V", 72)),
        },
        {
            "ill": (("I", 65), ("S", 12)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 5), ("V", 72)),
        },
        {
            "ill": (("I", 65), ("S", 12)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 5), ("V", 72)),
        },
        {
            "ill": (("I", 67), ("S", 10)),
            "aware": (("A", 69), ("UA", 8)),
            "vacc": (("UV", 5), ("V", 72)),
        },
    )


class TestMultiSpreading(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    def setUp(self) -> None:
        """Sets up most common testing parameters."""
        model, phenomenas, network = prepare_data()
        self.model = model
        self.phenomenas = phenomenas
        self.network = network

    def test_set_initial_states(self) -> None:
        """Checks if set_initial_states appends good statuses to nodes."""
        model = self.model
        network = self.network
        experiment = MultiSpreading(model, network)

        phenomenas_num = {
            "ill": (65, 10, 2),
            "aware": (60, 17),
            "vacc": (70, 7),
        }
        experiment.set_initial_states(phenomenas_num)

        # check if number of nodes with certain status is as expected
        for phenomena_name in self.phenomenas.keys():
            phenomena_graph = experiment._network.layers[phenomena_name]
            expected_nodes_states = {
                k: v
                for k, v in zip(
                    self.phenomenas[phenomena_name],
                    phenomenas_num[phenomena_name],
                )
            }
            real_nodes_states = {k: 0 for k in self.phenomenas[phenomena_name]}
            for node in phenomena_graph.nodes:
                status = phenomena_graph.nodes[node]["status"]
                real_nodes_states[status] += 1
            self.assertEqual(
                real_nodes_states,
                expected_nodes_states,
                f"Wrong number of states in layer {phenomena_name}: "
                f"expected {expected_nodes_states} found {real_nodes_states}",
            )

    def test_perform_propagation(self) -> None:
        """Checks if perform_propagation returns correct values."""
        experiment = MultiSpreading(self.model, self.network)
        initial_states = {
            "ill": (65, 10, 2),
            "aware": (60, 17),
            "vacc": (70, 7),
        }
        experiment.set_initial_states(initial_states)
        logs = experiment.perform_propagation(50)
        for k, v in logs._stats.items():
            for idx, row in v.iterrows():
                if idx == 0:
                    self.assertTrue(
                        np.all(row.values == initial_states[k]),
                        f"Particular states in initial epoch should be equal "
                        f"to given during defining an experiment"
                        f"{row.values} != {initial_states[k]}",
                    )
                else:
                    self.assertEqual(
                        row.sum(),
                        np.sum(initial_states[k]),
                        f"Sum of particular node states in epoch must be equal"
                        f" to the sum of all nodes in the network's layer "
                        f"sum({row.values}) != {np.sum(initial_states[k])}",
                    )


class TestExperimentLogger(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    def setUp(self) -> None:
        """Sets up most common testing parameters."""
        model, phenomenas, network = prepare_data()
        self.model = model
        self.phenomenas = phenomenas
        self.network = network
        experiment = MultiSpreading(self.model, self.network)
        initial_states = {
            "ill": (50, 27, 0),
            "aware": (30, 47),
            "vacc": (10, 67),
        }
        experiment.set_initial_states(initial_states)
        self.logs = experiment.perform_propagation(70)

    def test_plot(self) -> None:
        """Checks if visualisation is being stored."""
        with TemporaryDirectory() as out_dir:
            self.logs.plot(True, out_dir)
            self.assertTrue(
                "visualisation.png" in os.listdir(out_dir),
                f"After creating visualisation a gif file of name "
                f"visualisation.png in {out_dir} should be saved!",
            )

    def test_report(self) -> None:
        """Checks if report function writes out all files that it should."""
        with TemporaryDirectory() as out_dir:
            self.logs.report(visualisation=True, to_file=True, path=out_dir)
            exp_files = {
                "ill_propagation_report.csv",
                "model_report.txt",
                "network_report.txt",
                "visualisation.png",
                "vacc_propagation_report.csv",
                "aware_propagation_report.csv",
            }
            real_files = set(os.listdir(out_dir))
            self.assertEqual(
                real_files,
                exp_files,
                f"Report function should produce following files: {exp_files}, "
                f"but produced {real_files}",
            )

    def test__convert_logs(self) -> None:
        """Checks if logs convention is done properly."""
        raw_logs = prepare_logs()

        model_hyperparams = {
            "ill": ("S", "I", "R"),
            "aware": ("UA", "A"),
            "vacc": ("UV", "V"),
        }

        exp_ill = pd.DataFrame(
            {
                "S": {
                    0: 47,
                    1: 38,
                    2: 35,
                    3: 28,
                    4: 21,
                    5: 19,
                    6: 17,
                    7: 12,
                    8: 12,
                    9: 10,
                },
                "I": {
                    0: 30,
                    1: 39,
                    2: 42,
                    3: 49,
                    4: 56,
                    5: 58,
                    6: 60,
                    7: 65,
                    8: 65,
                    9: 67,
                },
                "R": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0,
                    9: 0,
                },
            }
        )
        exp_aware = pd.DataFrame(
            {
                "UA": {
                    0: 8,
                    1: 8,
                    2: 8,
                    3: 8,
                    4: 8,
                    5: 8,
                    6: 8,
                    7: 8,
                    8: 8,
                    9: 8,
                },
                "A": {
                    0: 69,
                    1: 69,
                    2: 69,
                    3: 69,
                    4: 69,
                    5: 69,
                    6: 69,
                    7: 69,
                    8: 69,
                    9: 69,
                },
            }
        )
        exp_vacc = pd.DataFrame(
            {
                "UV": {
                    0: 9,
                    1: 9,
                    2: 9,
                    3: 6,
                    4: 6,
                    5: 5,
                    6: 5,
                    7: 5,
                    8: 5,
                    9: 5,
                },
                "V": {
                    0: 68,
                    1: 68,
                    2: 68,
                    3: 71,
                    4: 71,
                    5: 72,
                    6: 72,
                    7: 72,
                    8: 72,
                    9: 72,
                },
            }
        )
        exp_logs_converted = {
            "ill": exp_ill,
            "aware": exp_aware,
            "vacc": exp_vacc,
        }

        logger = ExperimentLogger("model", "network")
        for i in raw_logs:
            logger._add_log(i)

        self.assertEqual(
            logger._stats,
            None,
            "Before convertion of logs field 'stat' should be empty",
        )

        logger._convert_logs(model_hyperparams)
        for phenomena, stat in logger._stats.items():
            self.assertTrue(
                exp_logs_converted[phenomena].equals(stat),
                f"Logs for phenomena {phenomena} incorrect. Expected value"
                f" {exp_logs_converted[phenomena]}, got {stat}",
            )

    def test__add_log(self) -> None:
        """Checks if logs are being colledded proprely during simulation."""
        raw_logs = prepare_logs()
        logger = ExperimentLogger("model", "network")

        lengths__raw_stats = [len(logger._raw_stats)]
        for log in raw_logs:
            logger._add_log(log)
            lengths__raw_stats.append(len(logger._raw_stats))

        exp_lengths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(
            lengths__raw_stats,
            exp_lengths,
            f"Lengths of '_raw_stats' field should be like {exp_lengths}, "
            f"got {lengths__raw_stats}.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
