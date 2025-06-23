# Copyright (c) 2025 by Mingshan Jia, Michał Czuba.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Functions for driver actor selection with local improvement."""

import multiprocessing
import multiprocessing.managers
import random
from typing import Any, Optional

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mds.greedy_search import (
    _minimum_dominating_set_with_initial,
)
from network_diffusion.mln.mds.utils import ShareableListManager
from network_diffusion.mln.mlnetwork import MultilayerNetwork


def get_mds_locimpr(
    net: MultilayerNetwork,
    timeout: Optional[float] = None,
    debug: bool = False,
) -> set[MLNetworkActor]:
    """
    Get driver actors for a network using greedy-based local improvement algo.

    The routine works as follows: (1) get the initial solution with
    `get_mds_greedy`, (2) try to improve the solution by iteratively trying to
    prune it.

    This method is inspired by a work by A Casado et al. "An iterated greedy
    algorithm for finding the minimum dominating set in graphs
    (https://doi.org/10.1016/j.matcom.2022.12.018) which was published in
    "Mathematics and Computers in Simulation", 2023, Volume 207.

    :param net: network to obtain minimal dominating set for
    :param timeout: a timeout for bigger networks, if none then it will be set
        in proportion of 5 minutes per 1000 actors
    :param debug: if true print debug statements
    :return: (sub)minimal dominating set
    """
    init_ds: set[Any] = set()
    for layer in net.layers:
        init_ds = _minimum_dominating_set_with_initial(net, layer, init_ds)
    if not timeout:
        timeout = net.get_actors_num() * 300 // 1000
    improv_ds = LocalImprovement(net, timeout, debug)(init_ds)
    return {net.get_actor(actor_id) for actor_id in improv_ds}


class LocalImprovement:
    """A class to prune initial Dominating Set."""

    def __init__(
        self, net: MultilayerNetwork, timeout: float, debug: bool = False
    ) -> None:
        """Initialise an object."""
        self.net = net
        self.actors = net.get_actors()
        self.timeout = timeout
        self.debug = debug

    def __call__(self, initial_set: set[Any]) -> set[Any]:
        """
        Perform the local improvement operation.

        The method uses a `ShareableListManager` to meet a timeout requirement.
        Namely, it tried to prune the `initial_set` for a given period of time
        and stores loccaly optimised solutions in the instance of a
        `ShareableListManager` class.

        :param initial_set: initial dominating set obtained with the greedy
            routine
        :return: pruned dominating set
        """
        with multiprocessing.managers.SharedMemoryManager() as smm:
            slm = ShareableListManager(smm, len(initial_set))
            slm.update_sl(list(initial_set))
            proc = multiprocessing.Process(
                target=self._local_improvement, args=(initial_set, slm)
            )
            proc.start()
            proc.join(self.timeout)
            if proc.is_alive():
                proc.terminate()
                print("Timeout reached, returning best-so-far solution.")
            return slm.get_as_pruned_set()

    def _local_improvement(
        self, initial_set: set[Any], final_set: ShareableListManager
    ) -> None:
        curr_dominating_set = set(initial_set)
        domination = self._compute_domination(curr_dominating_set)
        if self.debug:
            print(f"Current length of MDS: {len(initial_set)}")

        improvement = True
        while improvement:
            improvement = False

            # shuffle the dominating set to diversify search of neighbours
            curr_dominating_list = list(curr_dominating_set)
            random.shuffle(curr_dominating_list)

            for u in curr_dominating_list:

                # identify candidate replacements only which lead to a feasible
                # solution
                candidates_v = self._find_replacement_candidates(
                    u, curr_dominating_set, domination
                )
                random.shuffle(candidates_v)

                for v in candidates_v:

                    # store old solution for rollback if no improvement after
                    # checking
                    old_dominating_set = set(curr_dominating_set)

                    # attempt the exchange move
                    new_dominating_set = (curr_dominating_set - {u}) | {v}
                    if self._is_feasible(new_dominating_set):

                        # after a feasible exchange remove redundancies
                        reduced_set = self._remove_redundant_vertices(
                            new_dominating_set
                        )

                        # check if we actually improved DS (i.e., reduced the
                        # size of the solution)
                        if len(reduced_set) < len(old_dominating_set):

                            # if so update domination and break
                            curr_dominating_set = reduced_set
                            final_set.update_sl(list(curr_dominating_set))
                            domination = self._compute_domination(
                                curr_dominating_set
                            )
                            improvement = True
                            break

                        # if no improvement after redundancy removal, revert to
                        # the old solution
                        curr_dominating_set = old_dominating_set

                        if self.debug:
                            print(f"Curr MDS len: {len(curr_dominating_set)}")

                # restart the outer loop after finding the first improvement,
                # otherwise exit the function.
                if improvement:
                    break

    def _compute_domination(
        self, dominating_set: set[Any]
    ) -> dict[str, dict[Any, set[Any]]]:
        """
        Compute the domination map for the current dominating set per layer.

        Return a dictionary where keys are layer names and values are
        dictionaries mapping node IDs to sets of dominators in that layer.
        """
        domination_map: dict[str, dict[Any, set[Any]]] = {
            layer: {actor.actor_id: set() for actor in self.actors}
            for layer in self.net.layers
        }
        for l_name, l_graph in self.net.layers.items():
            for actor_id in dominating_set:
                if actor_id in l_graph.nodes:
                    domination_map[l_name][actor_id].add(
                        actor_id
                    )  # a node dominates itself
                    for neighbour in l_graph[actor_id]:
                        domination_map[l_name][neighbour].add(actor_id)
        return domination_map

    def _get_excusevely_dominated_by_u(
        self, u: Any, domination: dict[str, dict[Any, set[Any]]]
    ) -> dict[str, set[Any]]:
        """Get nodes that are exclusevely dominated by `u` in the network."""
        ed = {}
        for layer, net_layer in self.net.layers.items():
            if u in net_layer:
                ed[layer] = {
                    w
                    for w in set(net_layer[u]) | {u}
                    if domination[layer][w] == {u}
                }
            else:  # no nodes are exclusively dominated by u in this layer
                ed[layer] = set()
        return ed

    def _find_replacement_candidates(
        self,
        u: Any,
        dominating_set: set[Any],
        domination: dict[str, dict[Any, set[Any]]],
    ) -> list[Any]:
        """
        Find candidate nodes v that can replace u in the dominating set.

        Also ensure that all layers remain dominated.
        """
        exclusively_dominated = self._get_excusevely_dominated_by_u(
            u=u,
            domination=domination,
        )

        # find valid replacement candidates
        candidates = []
        for v in [x.actor_id for x in self.actors]:
            if v in dominating_set:
                continue

            # ensure v exists in all layers where exclusively dominated
            # nodes are expected
            if all(
                v in self.net.layers[layer]
                and nodes.issubset(set(self.net.layers[layer][v]) | {v})
                for layer, nodes in exclusively_dominated.items()
            ):
                candidates.append(v)

        return candidates

    def _is_feasible(self, dominating_set: set[Any]) -> bool:
        """Check if the dominating set is feasible across all layers."""
        for _, l_graph in self.net.layers.items():
            dominated = set()
            for actor_id in dominating_set:
                if actor_id in l_graph.nodes:
                    dominated.add(actor_id)
                    dominated.update(l_graph[actor_id])
            if dominated != set(l_graph.nodes()):
                return False
        return True

    def _remove_redundant_vertices(self, dominating_set: set[Any]) -> set[Any]:
        """
        Try to remove redundant vertices from DS without losing feasibility.

        A vertex is redundant if removing it still leaves all nodes dominated.
        Returns a new dominating set with as many redundant vertices removed as
        possible. We'll attempt to remove vertices one by one. A simple
        (although not necessarily minimum) approach is to try removing each
        vertex and see if the set remains feasible. If yes, then permanently
        remove it.
        """
        improved_set = set(dominating_set)
        under_improvement = True
        while under_improvement:
            under_improvement = False
            for d in improved_set:
                candidate_set = improved_set - {d}
                if self._is_feasible(candidate_set):
                    improved_set = candidate_set
                    under_improvement = True
                    break  # break to re-check from scratch after every removal
        return improved_set
