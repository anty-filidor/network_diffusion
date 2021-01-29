#!/usr/bin/env python3

import os
import random
import shutil
import string
import unittest

import networkx as nx

from network_diffusion import FlatSpreading


class TestFlatSpreadingSI(unittest.TestCase):
    """Integration test class for SI spreading."""

    def setUp(self) -> None:
        """Sets up testing parameters and performs simulation."""
        self.exp_M = nx.les_miserables_graph()
        self.exp_beta = 0.1
        self.exp_I = 0.05
        self.exp_name = "".join(random.choices(string.ascii_letters, k=5))
        self.out_dir = "".join(random.choices(string.ascii_letters, k=5))

        (
            self.list_S,
            self.list_I,
            self.list_iter,
            self.nodes_infected,
            self.par,
        ) = FlatSpreading.si_diffusion(
            self.exp_M, self.exp_I, self.exp_beta, name=self.exp_name
        )

    def test_output_lengths(self) -> None:
        """Test if lengths of output numerical values are the same."""
        msg = (
            f"Lengths are not equal! list_S={len(self.list_S)}, "
            f"list_I={len(self.list_I)}, list_iter={len(self.list_iter)},"
            f" nodes_infected={len(self.nodes_infected)}"
        )
        self.assertTrue(
            len(self.list_S)
            == len(self.list_I)
            == len(self.list_iter)
            == len(self.nodes_infected),
            msg,
        )

    def test_simulation_coherency(self) -> None:
        """Tests if list_S[n]+list_I[n]==len(exp_M) in each n (epoch)."""
        msg = "Number of infected and suspected nodes in epoch doesn't match"
        for n in self.list_iter:
            self.assertTrue(
                self.list_S[n] + self.list_I[n] == len(self.exp_M),
                f"{msg}, in epoch {n}: S={self.list_S[n]}, I={self.list_I[n]},"
                f" total_nodes={len(self.exp_M)}",
            )

    def test_infections_coherency(self) -> None:
        """Tests if names of new infected nodes in each epoch=numeric stats."""
        msg = (
            "Number of new infected nodes in epoch should be equal to "
            "length of it's counterparts names in nodes_infected list"
        )
        sum = 0
        for n in self.list_iter:
            self.assertTrue(
                len(self.nodes_infected[n]) == self.list_I[n] - sum,
                f"{msg}, in epoch {n}: nodes_infected={self.nodes_infected[n]},"
                f" expected length={self.list_I[n]-sum}",
            )
            sum = self.list_I[n]

    def test_output_input_params_coherency(self) -> None:
        """Tests if experiment params returned by simulator == input params."""
        self.assertEqual(
            self.par[0],
            self.exp_name,
            f"Experiment names: {self.par[0]}!={self.exp_name}",
        )
        self.assertEqual(
            self.par[1],
            self.exp_beta,
            f"Beta coefficient: {self.par[1]}!={self.exp_beta}",
        )
        self.assertEqual(
            self.par[2],
            self.exp_I,
            f"I fractions: {self.par[2]}!={self.exp_I}",
        )

    def test_compare_high_beta(self) -> None:
        """Checks if change of beta parameter changes duration of simulat."""
        beta = 1
        comp_out_vals = FlatSpreading.si_diffusion(
            self.exp_M, self.exp_I, beta
        )
        self.assertGreater(
            len(self.list_S),
            len(comp_out_vals[0]),
            f"Simulation logs for beta={beta} should be greater than for beta"
            f"={self.exp_beta} ({len(self.list_S)}!>{len(comp_out_vals[0])})",
        )

    def test_compare_high_I(self) -> None:
        """Checks if change of I parameter changes duration of simulation."""
        new_exp_I = 1
        comp_out_vals = FlatSpreading.si_diffusion(
            self.exp_M, new_exp_I, self.exp_beta
        )
        for i in comp_out_vals[:-1]:
            self.assertEqual(
                len(i),  # type: ignore
                1,
                f"Simulation logs for I={new_exp_I} should be "
                f"equal to 1 (len({i})!=1)",
            )

    def test_visualise_si_node(self) -> None:
        """Checks if visualisation is being generated."""
        FlatSpreading.visualise_si_nodes(
            self.exp_M, self.nodes_infected, self.par, self.out_dir
        )
        vis_si_nodes_name = f"{self.exp_name}_si_n.gif"
        self.assertTrue(
            vis_si_nodes_name in os.listdir(self.out_dir),
            f"After creating visualisation a gif file of name "
            f"{vis_si_nodes_name} should be saved!",
        )
        shutil.rmtree(self.out_dir)

    def test_visualise_si_nodes_edges(self) -> None:
        """Checks if visualisation is being generated."""
        FlatSpreading.visualise_si_nodes_edges(
            self.exp_M, self.nodes_infected, self.par, self.out_dir
        )
        vis_si_nodes_edges_name = f"{self.exp_name}_si_ne.gif"
        self.assertTrue(
            vis_si_nodes_edges_name in os.listdir(self.out_dir),
            f"After creating visualisation a gif file of name "
            f"{vis_si_nodes_edges_name} should be saved!",
        )
        shutil.rmtree(self.out_dir)


class TestFlatSpreadingSIR(unittest.TestCase):
    """Integration test class for SIR spreading."""

    def setUp(self) -> None:
        """Sets up testing parameters and performs simulation."""
        self.exp_M = nx.les_miserables_graph()
        self.exp_beta = 0.1
        self.exp_gamma = 0.2
        self.exp_I = 0.05
        self.exp_name = "".join(random.choices(string.ascii_letters, k=5))
        self.out_dir = "".join(random.choices(string.ascii_letters, k=5))

        (
            self.list_S,
            self.list_I,
            self.list_R,
            self.list_iter,
            self.nodes_infected,
            self.nodes_recovered,
            self.par,
        ) = FlatSpreading.sir_diffusion(
            self.exp_M,
            self.exp_I,
            self.exp_beta,
            self.exp_gamma,
            name=self.exp_name,
        )

    def test_output_lengths(self) -> None:
        """Test if lengths of output numerical values are the same."""
        msg = (
            f"Lengths are not equal! list_S={len(self.list_S)}, "
            f"list_I={len(self.list_I)}, list_R={len(self.list_R)}, "
            f"list_iter={len(self.list_iter)}, "
            f"nodes_infected={len(self.nodes_infected)}, "
            f"nodes_recovered={len(self.nodes_recovered)}"
        )
        self.assertTrue(
            len(self.list_R)
            == len(self.list_S)
            == len(self.list_I)
            == len(self.list_iter)
            == len(self.nodes_infected)
            == len(self.nodes_recovered),
            msg,
        )

    def test_simulation_coherency(self) -> None:
        """Tests list_S[n]+list_I[n]+list_R[n]==len(exp_M) in each epoch-n."""
        msg = "Number of infec, susp, recov nodes in epoch doesn't match"
        for n in self.list_iter:
            self.assertTrue(
                self.list_S[n] + self.list_I[n] + self.list_R[n]
                == len(self.exp_M),
                f"{msg}, in epoch {n}: S={self.list_S[n]}, I={self.list_I[n]},"
                f" I={self.list_R[n]}, total_nodes={len(self.exp_M)}",
            )

    def test_infections_coherency(self) -> None:
        """Tests if names of new infec. nodes in each epoch ~ numeric stats."""
        msg = (
            "Number of new infected nodes in epoch should be <= to length "
            "of all infected nodes in network in epoch"
        )
        for n in self.list_iter:
            self.assertTrue(
                len(self.nodes_infected[n]) <= self.list_I[n],
                f"{msg}, in epoch {n}: all nodes infected="
                f"{self.nodes_infected[n]}, fresh newly infected nodes="
                f"{self.list_I[n]}",
            )

    def test_recoveries_coherency(self) -> None:
        """Tests if names of new recov. nodes in each epoch ~ numeric stats."""
        msg = (
            "Number of new recovered nodes in epoch should be <= to "
            "length of all recovered nodes in network in epoch"
        )
        for n in self.list_iter:
            self.assertTrue(
                len(self.nodes_recovered[n]) <= self.list_R[n],
                f"{msg}, in epoch {n}: all nodes infected="
                f"{self.nodes_recovered[n]}, fresh newly recovered nodes="
                f"{self.list_R[n]}",
            )

    def test_output_input_params_coherency(self) -> None:
        """Tests if experiment params returned by simulator == input params."""
        self.assertEqual(
            self.par[0],
            self.exp_name,
            f"Experiment names: {self.par[0]}!={self.exp_name}",
        )
        self.assertEqual(
            self.par[1],
            self.exp_beta,
            f"Beta coefficient: {self.par[1]}!={self.exp_beta}",
        )
        self.assertEqual(
            self.par[2],
            self.exp_gamma,
            f"Gamma coefficient: {self.par[2]}!={self.exp_gamma}",
        )
        self.assertEqual(
            self.par[3],
            self.exp_I,
            f"I fractions: {self.par[3]}!={self.exp_I}",
        )

    def test_visualise_si_node(self) -> None:
        """Checks if visualisation is being generated."""
        FlatSpreading.visualise_sir_nodes(
            self.exp_M,
            self.nodes_infected,
            self.nodes_recovered,
            self.par,
            self.out_dir,
        )
        vis_sir_nodes_name = f"{self.exp_name}_sir_n.gif"
        self.assertTrue(
            vis_sir_nodes_name in os.listdir(self.out_dir),
            f"After creating visualisation a gif file of name "
            f"{vis_sir_nodes_name} should be saved!",
        )
        shutil.rmtree(self.out_dir)

    def test_visualise_si_nodes_edges(self) -> None:
        """Checks if visualisation is being generated."""
        FlatSpreading.visualise_sir_nodes_edges(
            self.exp_M,
            self.nodes_infected,
            self.nodes_recovered,
            self.par,
            self.out_dir,
        )
        vis_sir_nodes_edges_name = f"{self.exp_name}_sir_ne.gif"
        self.assertTrue(
            vis_sir_nodes_edges_name in os.listdir(self.out_dir),
            f"After creating visualisation a gif file of name "
            f"{vis_sir_nodes_edges_name} should be saved!",
        )
        shutil.rmtree(self.out_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
