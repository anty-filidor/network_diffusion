import os
import unittest
from tempfile import TemporaryDirectory

import networkx as nx
import pandas as pd

from network_diffusion import MultilayerNetwork, Simulator
from network_diffusion.models import DSAAModel
from network_diffusion.simulator import Logger
from network_diffusion.tests.models.test_dsaa_model import prepare_compartments


def prepare_logs():
    """Prepare logs for tests of Logger class."""
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


class TestLogger(unittest.TestCase):
    """Test class for MultilayerNetwork class."""

    def setUp(self):
        """Set up most common testing parameters."""
        compartments, phenomena = prepare_compartments()
        self.phenomena = phenomena

        # init multilayer network from nx predefined network
        network = MultilayerNetwork.from_nx_layer(
            nx.les_miserables_graph(), [*phenomena.keys()]
        )
        self.network = network

        # init model
        self.model = DSAAModel(compartments)
        self.model.compartments.seeding_budget = {
            "ill": (65, 35, 0),
            "aware": (39, 61),
            "vacc": (13, 87),
        }

        experiment = Simulator(self.model, self.network)
        self.logs = experiment.perform_propagation(70)

    def test_plot(self):
        """Check if visualisation is being stored."""
        with TemporaryDirectory() as out_dir:
            self.logs.plot(True, out_dir)
            self.assertTrue(
                "visualisation.png" in os.listdir(out_dir),
                f"After creating visualisation a gif file of name "
                f"visualisation.png in {out_dir} should be saved!",
            )

    def test_report(self):
        """Check if report function writes out all files that it should."""
        with TemporaryDirectory() as out_dir:
            self.logs.report(visualisation=True, path=out_dir)
            exp_files = {
                "ill_propagation_report.csv",
                "model_report.txt",
                "network_report.txt",
                "visualisation.png",
                "vacc_propagation_report.csv",
                "aware_propagation_report.csv",
                "local_stats.json",
            }
            real_files = set(os.listdir(out_dir))
            self.assertEqual(
                real_files,
                exp_files,
                f"Report function should produce following files: {exp_files},"
                f" but produced {real_files}",
            )

    def test__convert_logs(self):
        """Check if logs convention is done properly."""
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

        logger = Logger("model", "network")
        for i in raw_logs:
            logger.add_global_stat(i)

        self.assertEqual(
            logger._global_stats_converted,
            {},
            "Before convertion of logs field 'stat' should be empty",
        )

        logger.convert_logs(model_hyperparams)
        for phenomena, stat in logger._global_stats_converted.items():
            pd.testing.assert_frame_equal(
                exp_logs_converted[phenomena], stat, check_dtype=False
            )

    def test__add_log(self):
        """Check if logs are being colledded proprely during simulation."""
        raw_logs = prepare_logs()
        logger = Logger("model", "network")

        lengths__raw_stats = [len(logger._global_stats)]
        for log in raw_logs:
            logger.add_global_stat(log)
            lengths__raw_stats.append(len(logger._global_stats))

        exp_lengths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(
            lengths__raw_stats,
            exp_lengths,
            f"Lengths of '_raw_stats' field should be like {exp_lengths}, "
            f"got {lengths__raw_stats}.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
