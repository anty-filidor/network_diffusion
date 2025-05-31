"""Script with functions for driver actor selections with local improvement."""

import multiprocessing
import multiprocessing.managers
import multiprocessing.shared_memory
import random
import time
from typing import Any

from src.models.mds.greedy_search import minimum_dominating_set_with_initial
from src.models.mds.utils import ShareableListManager

import network_diffusion as nd

# try:
#     import sys
#     from pathlib import Path
#     import src
# except:
#     sys.path.append(str(Path(__file__).parent.parent.parent.parent))
#     print(sys.path)


def get_mds_locimpr(
    net: nd.MultilayerNetwork, timeout: int = None, debug: bool = False
) -> list[nd.MLNetworkActor]:
    """Return driver actors for a given network using MDS and local improvement."""
    # step 1: compute initial Minimum Dominating Set
    initial_dominating_set: set[Any] = set()
    for layer in net.layers:
        initial_dominating_set = minimum_dominating_set_with_initial(
            net, layer, initial_dominating_set
        )

    # step 2: apply Local Improvement to enhance the Dominating Set
    if not timeout:
        timeout = (
            net.get_actors_num() * 300 // 1000
        )  # proportion is 5 minutes per 1000 actors
    improved_dominating_set = LocalImprovement(net, timeout, debug)(
        initial_dominating_set
    )

    return [net.get_actor(actor_id) for actor_id in improved_dominating_set]


class LocalImprovement:
    """A class to prune initial Dominating Set."""

    def __init__(
        self, net: nd.MultilayerNetwork, timeout: float, debug: bool = False
    ):
        self.net = net
        self.actors = net.get_actors()
        self.timeout = timeout
        self.debug = debug

    def __call__(self, initial_set: set[Any]) -> set[Any]:
        with multiprocessing.managers.SharedMemoryManager() as smm:
            slm = ShareableListManager(smm, len(initial_set))
            slm.sl = list(initial_set)
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
        """Perform local improvement on the initial DS using the First Improvement strategy."""
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

                # identify candidate replacements only which lead to a feasible solution
                candidates_v = self._find_replacement_candidates(
                    u, curr_dominating_set, domination
                )
                random.shuffle(candidates_v)

                for v in candidates_v:

                    # store old solution for rollback if no improvement after checking
                    old_dominating_set = set(curr_dominating_set)

                    # attempt the exchange move
                    new_dominating_set = (curr_dominating_set - {u}) | {v}
                    if self._is_feasible(new_dominating_set):

                        # after a feasible exchange remove redundancies
                        reduced_set = self._remove_redundant_vertices(
                            new_dominating_set
                        )

                        # check if we actually improved (reduced the size of the solution)
                        if len(reduced_set) < len(old_dominating_set):

                            # if so update domination and break
                            curr_dominating_set = reduced_set
                            final_set.sl = list(curr_dominating_set)
                            domination = self._compute_domination(
                                curr_dominating_set
                            )
                            improvement = True
                            break

                        # if no improvement after redundancy removal, revert to old solution
                        else:
                            curr_dominating_set = old_dominating_set

                        if self.debug:
                            print(
                                f"Current length of MDS: {len(curr_dominating_set)}"
                            )

                # restart the outer loop after finding the first improvement, otherwise exit funct.
                if improvement:
                    break

    def _compute_domination(
        self, dominating_set: set[Any]
    ) -> dict[str, dict[Any, set[Any]]]:
        """
        Compute the domination map for the current dominating set per layer.

        Return a dictionary where keys are layer names and values are dictionaries mapping node IDs
        to sets of dominators in that layer.
        """
        domination_map = {
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
        """Get nodes that are exclusevely dominated by node u in the network."""
        ed = {}
        for layer, net_layer in self.net.layers.items():
            if u in net_layer:
                ed[layer] = {
                    w
                    for w in set(net_layer[u]) | {u}
                    if domination[layer][w] == {u}
                }
            else:
                ed[layer] = (
                    set()
                )  # nNo nodes exclusively dominated by u in this layer
        return ed

    def _find_replacement_candidates(
        self,
        u: Any,
        dominating_set: set[Any],
        domination: dict[str, dict[Any, set[Any]]],
    ) -> list[Any]:
        """
        Find candidate nodes v that can replace u in the dominating set, ensuring that all layers
        remain dominated.
        """
        exclusively_dominated = self._get_excusevely_dominated_by_u(
            u, domination
        )

        # find valid replacement candidates
        candidates = []
        for v in [x.actor_id for x in self.actors]:
            if v in dominating_set:
                continue

            # ensure v exists in all layers where exclusively dominated nodes are expected
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

    def _remove_redundant_vertices(self, dominating_set: set[Any]) -> set[any]:
        """
        Try to remove redundant vertices from the dominating_set without losing feasibility.

        A vertex is redundant if removing it still leaves all nodes dominated.
        Returns a new dominating set with as many redundant vertices removed as possible.
        We'll attempt to remove vertices one by one. A simple (although not necessarily minimum)
        approach is to try removing each vertex and see if the set remains feasible. If yes,
        permanently remove it.
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


if __name__ == "__main__":
    from src.loaders.net_loader import load_network
    from src.models.mds.greedy_search import get_mds_greedy
    from utils import is_dominating_set

    # net = load_network("sf2", as_tensor=False)
    net = load_network("ckm_physicians", as_tensor=False)

    start_time = time.time()
    mds = get_mds_locimpr(net, debug=True)
    # mds = get_mds_greedy(net)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.2f} seconds")

    # mds.pop()
    if is_dominating_set(candidate_ds=mds, network=net):
        print(
            f"A {len(mds)}-length set: {set(ac.actor_id for ac in mds)} is dominating!"
        )
    else:
        print(
            f"A {len(mds)}-length set: {set(ac.actor_id for ac in mds)} is not dominating!"
        )
