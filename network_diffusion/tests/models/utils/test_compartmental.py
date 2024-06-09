#!/usr/bin/env python3
# pylint: disable-all
# type: ignore

"""Tests for the network_diffusion.propagation_model."""

import unittest

import networkx as nx

from network_diffusion.models.utils.compartmental import CompartmentalGraph
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


def get_compiled_model():
    """Prepare compiled propagation model for tests."""
    model = CompartmentalGraph()
    states = [["A", "B", "C"], ["A", "B"], ["A", "B"]]
    processes = ("1", "2", "3")
    for proc, phenom in zip(processes, states):
        model.add(proc, phenom)
    model.compile(background_weight=0.005)
    return model


class TestCompartmentalGraph(unittest.TestCase):
    """Test class for CompartmentalGraph class."""

    def test_add(self):
        """Test if function adds process to the model."""
        model = CompartmentalGraph()
        model.add("1", ["A", "B", "C"])
        self.assertEqual(
            model.__dict__,
            {
                "graph": {},
                "background_weight": float("inf"),
                "_seeding_budget": {},
                "1": ["A", "B", "C"],
            },
            "add func seems to have no effect",
        )

    def test__str__empty(self):
        """Check describing a model that not have been initialised."""
        self.maxDiff = None
        log = (
            "============================================\n"
            + "compartmental model\n"
            + "--------------------------------------------\n"
            + "processes, their states and initial sizes:\n"
            + "--------------------------------------------\n"
            + "process '1' transitions with nonzero weight:\n\t"
            + "from A to B with probability 0.2137 and constrains []\n\t"
            + "from B to A with probability 0.2137 and constrains []\n"
            + "============================================\n"
        )

        model = CompartmentalGraph()
        model.add("1", ["A", "B"])
        model.compile(background_weight=0.2137)
        self.assertEqual(
            model._get_desctiprion_str(),
            log,
            "Empty description string not in expected form",
        )

    def test_get_model_hyperparams(self):
        """Check if get_model_hyperparams function behaves correctly."""
        model = get_compiled_model()
        self.assertEqual(
            {"1": ["A", "B", "C"], "2": ["A", "B"], "3": ["A", "B"]},
            model.get_compartments(),
            "Incorrect hyperparameters of CompartmentalGraph!",
        )

    def test_compile(self):
        """Check if compilation runs correctly."""
        model = CompartmentalGraph()
        model.add("1", ["A", "B", "C"])
        model.add("2", ["A", "B"])

        self.assertEqual(
            len(model.graph),
            0,
            msg="Model before compilation should have empty graph field",
        )

        model.compile(background_weight=0.1)
        self.assertEqual(
            [*model.graph.keys()],
            ["1", "2"],
            "After compilation keys should be like ['1', '2']",
        )

        exp_graph_phenomena_1 = nx.DiGraph(
            [
                (("1.A", "2.A"), ("1.B", "2.A")),
                (("1.A", "2.A"), ("1.C", "2.A")),
                (("1.B", "2.A"), ("1.A", "2.A")),
                (("1.B", "2.A"), ("1.C", "2.A")),
                (("1.C", "2.A"), ("1.A", "2.A")),
                (("1.C", "2.A"), ("1.B", "2.A")),
                (("1.A", "2.B"), ("1.B", "2.B")),
                (("1.A", "2.B"), ("1.C", "2.B")),
                (("1.B", "2.B"), ("1.A", "2.B")),
                (("1.B", "2.B"), ("1.C", "2.B")),
                (("1.C", "2.B"), ("1.A", "2.B")),
                (("1.C", "2.B"), ("1.B", "2.B")),
            ]
        )
        exp_graph_phenomena_2 = nx.DiGraph(
            [
                (("1.A", "2.A"), ("1.A", "2.B")),
                (("1.A", "2.B"), ("1.A", "2.A")),
                (("1.B", "2.A"), ("1.B", "2.B")),
                (("1.B", "2.B"), ("1.B", "2.A")),
                (("1.C", "2.A"), ("1.C", "2.B")),
                (("1.C", "2.B"), ("1.C", "2.A")),
            ]
        )
        self.assertEqual(
            nx.is_isomorphic(model.graph["1"], exp_graph_phenomena_1),
            True,
            f"Graph in layer 1 should be like {exp_graph_phenomena_1}",
        )
        self.assertEqual(
            nx.is_isomorphic(model.graph["2"], exp_graph_phenomena_2),
            True,
            f"Graph in layer 2 should be like {exp_graph_phenomena_2}",
        )

    def test_set_transition_canonical(self):
        """Checks if setting transitions in canonical way is possible."""
        model = get_compiled_model()
        weight = model.graph["2"][("1.C", "2.A", "3.B")][("1.C", "2.B", "3.B")]
        self.assertEqual(
            weight["weight"],
            0.005,
            f"Transition {('1.C', '2.A', '3.B'), ('1.C', '2.B', '3.A')} "
            f"after setting weight should value {0.005}, but it has"
            f" {weight['weight']}",
        )

        new_weight = 0.2137
        model.set_transition_canonical(
            "2",
            (
                ("1.C", "2.A", "3.B"),
                ("1.C", "2.B", "3.B"),
            ),
            new_weight,
        )

        weight = model.graph["2"][("1.C", "2.A", "3.B")][("1.C", "2.B", "3.B")]
        self.assertEqual(
            weight["weight"],
            new_weight,
            f"Transition {('1.C', '2.A', '3.B'), ('1.C', '2.B', '3.A')} "
            f"after setting weight should value {new_weight}, but it has"
            f" {weight['weight']}",
        )

    def test_set_transition_fast(self):
        """Checks if setting transitions in fast way is possible."""
        model = get_compiled_model()
        weight = model.graph["1"][("1.C", "2.B", "3.A")][("1.A", "2.B", "3.A")]
        self.assertEqual(
            weight["weight"],
            0.005,
            f"Transition {('1.C', '2.B', '3.A'), ('1.A', '2.B', '3.A')} "
            f"before setting weight should have it's background value (0.005),"
            f" but it has {weight['weight']}",
        )

        new_weight = 0.1791
        model.set_transition_fast("1.C", "1.A", ("2.B", "3.A"), new_weight)
        weight = model.graph["1"][("1.C", "2.B", "3.A")][("1.A", "2.B", "3.A")]
        self.assertEqual(
            weight["weight"],
            new_weight,
            f"Transition {('1.C', '2.B', '3.A'), ('1.A', '2.B', '3.A')} "
            f"after setting weight should value {new_weight}, but it has "
            f"{weight['weight']}",
        )

    def test_set_transitions_in_random_edges(self):
        """Check if setting transitions in random way is possible."""
        model = get_compiled_model()
        layer_1_w = {0.4: 0, 0.5: 0}
        layer_2_w = {0.3: 0, 0.2: 0, 0.1: 0}
        model.set_transitions_in_random_edges(
            [layer_1_w.keys(), layer_2_w.keys(), {}]
        )

        for i in ["1", "2"]:
            for edge in model.graph[i].edges:
                weigth = model.graph[i].edges[edge]["weight"]
                if weigth in layer_1_w.keys():
                    layer_1_w[weigth] += 1
                elif weigth in layer_2_w.keys():
                    layer_2_w[weigth] += 1

        for i in layer_1_w:
            self.assertEqual(
                layer_1_w[i],
                1,
                f"Weight {i} in layer 1 set up "
                + f"{layer_1_w[i]} times (expected 1).",
            )
        for i in layer_2_w:
            self.assertEqual(
                layer_2_w[i],
                1,
                f"Weight {i} in layer 2 set up "
                + f"{layer_2_w[i]} times (expected 1).",
            )

    def test_get_possible_transitions(self):
        """Check if possible transforms are gave correctly."""
        model = get_compiled_model()
        case_1 = model.get_possible_transitions(
            state=("1.C", "2.B", "3.A"), layer="1"
        )
        exp_res_1 = {"A": 0.005, "B": 0.005}
        self.assertEqual(
            case_1,
            exp_res_1,
            f"Possible transforms in node (1.C, 2.B, 3.A) in layer 1 should "
            f"be {exp_res_1}, was {case_1}",
        )

        case_2 = model.get_possible_transitions(
            state=("1.C", "2.B", "3.A"), layer="2"
        )
        exp_res_2 = {"A": 0.005}
        self.assertEqual(
            case_2,
            exp_res_2,
            f"Possible transforms in node (1.C, 2.B, 3.A) in layer 1 should "
            f"be {exp_res_2}, was {case_2}",
        )

        case_3 = model.get_possible_transitions(
            state=("1.C", "2.B", "3.A"), layer="3"
        )
        exp_res_3 = {"B": 0.005}
        self.assertEqual(
            case_3,
            exp_res_3,
            f"Possible transforms in node (1.C, 2.B, 3.A) in layer 3 should "
            f"be {exp_res_3}, was {case_3}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
