# Copyright 2020 by Michał Czuba, Piotr Bródka. All Rights Reserved.
#
# This file is part of Network Diffusion.
#
# Network Diffusion is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Network Diffusion is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# Network Diffusion. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Functions for the propagation model definition."""

# pylint: disable=W0141

import itertools
from random import choice
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


class PropagationModel:
    """Class which encapsulates model of processes speared in network."""

    def __init__(self) -> None:
        """Create empty object."""
        self.graph: Dict[str, nx.Graph] = {}
        self.background_weight: float = float("inf")

    def add(self, layer_name: str, layer_type: List[str]) -> None:
        """
        Add propagation model for the given layer.

        :param layer: name of network's layer
        :param type: type of model, i.e. names of states like ['s', 'i', 'r']
        """
        assert len(layer_type) == len(
            set(layer_type)
        ), f"Phenomena names {layer_type} must be unique!"
        assert layer_name != "graph", 'Layer cannot be named "graph"'
        assert (
            layer_name != "background_weight"
        ), 'Layer cannot be named "background_weight"'
        self.__setattr__(layer_name, layer_type)

    def describe(
        self, to_print: bool = True, full_graph: bool = False
    ) -> Optional[str]:
        """
        Print out parameters of the object.

        :param to_print: a flag, if true method prints out description to
            console, otherwise it returns string
        :param full_graph: a flag, if true method prints out all propagation
            model states with those with 0 weight

        :return: if to_print == True returns string describing object,
            otherwise returns None, but prints out description to the console
        """
        dscrpt_str = self._get_description_str(full_graph)
        if not to_print:
            return dscrpt_str
        else:
            print(dscrpt_str)
            return None

    def _get_description_str(self, full_graph: bool = False) -> str:
        """
        Get string describing the object.

        :param full_graph: a flag, if true method prints out all propagation
            model states with those with 0 weight

        :return: string describing object
        """
        # pylint: disable=R1702, R1719
        detailed_str = ""
        basic_str = (
            "============================================\n"
            "model of propagation\n"
            "--------------------------------------------\n"
        )
        basic_str += "phenomenas and their states:"
        for name, val in self.__dict__.items():
            if name == "graph":
                if len(self.graph) == 0:
                    basic_str += f"\n\t{name}: not initialised\n"
                    continue
                detailed_str += "\n"
                for g_name, g_net in val.items():
                    if full_graph:
                        detailed_str += f"\n{g_name} transitions: "
                        detailed_str += f"{g_net.edges().data()}\n"
                    else:
                        detailed_str += (
                            f"\nlayer '{g_name}' transitions with nonzero "
                            f"probability:\n"
                        )
                        for edge in g_net.edges():
                            if g_net.edges[edge]["weight"] == 0.0:
                                continue
                            mask = [
                                True if g_name in _ else False for _ in edge[0]
                            ]
                            start = np.array(edge[0])[mask][0].split(".")[1]
                            finish = np.array(edge[1])[mask][0].split(".")[1]
                            constraints = np.array(edge[0])[
                                [not _ for _ in mask]
                            ]
                            weight = g_net.edges[edge]["weight"]
                            detailed_str += (
                                f"\tfrom {start} to {finish} with "
                                f"probability {weight} and constrains "
                                f"{constraints}\n"
                            )
            elif name == "background_weight":
                continue
            else:
                basic_str += f"\n\t{name}: {val}"

        detailed_str += "============================================"

        return basic_str + detailed_str

    def get_model_hyperparams(self) -> Dict[str, Tuple[Dict[str, Any]]]:
        """
        Get model parameters, i.e. names of layers and states in each layer.

        :return: dictionary keyed by names of layer, valued by tuples of
            states labels
        """
        hyperparams = {}
        for name, val in self.__dict__.items():
            if name not in {"graph", "background_weight"}:
                hyperparams[name] = val
        return hyperparams

    def compile(
        self, background_weight: float = 0.0, track_changes: bool = False
    ) -> None:
        """
        Create transition matrices for models of propagation in each layer.

        All transition probabilities are set to 0. To be more specific,
        transitions matrices are stored as a networkx one-directional graph.
        After compilation user is able to set certain transitions in model.

        :param background_weight: [0,1] describes default weight of transition
            to make propagation more realistic by default it is set to 0
        :param track_changes: a flag to track progress of matrices creation
        """
        # pylint: disable=R0914
        assert 1 >= background_weight >= 0, "Weight value not in [0, 1] range"

        # initialise dictionary to store transition graphs for each later
        transitions_graphs = {}

        # create transition graph for each layer
        for layer_name, layer_type in self.__dict__.items():
            if layer_name in {"graph", "background_weight"}:
                continue
            if track_changes:
                print(layer_name)

            # prepare names in current layer
            cl_names = [str(layer_name) + "." + str(v) for v in layer_type]
            if track_changes:
                print("crtlr", cl_names)

            # copy layers description and delete current layer to prepare
            # constants for current layer
            self_dict_copy = self.__dict__.copy()
            del self_dict_copy[layer_name]
            del self_dict_copy["graph"]
            del self_dict_copy["background_weight"]

            # prepare names in other layers which are constant to current layer
            ol_names = []
            for i in self_dict_copy.keys():
                i_names = [str(i) + "." + str(v) for v in self_dict_copy[i]]
                ol_names.append(i_names)

            # prepare constants
            product = [*itertools.product(*ol_names)]
            if track_changes:
                print("const", product)

            # crate transitions graph from product and names of states in
            # current layer
            graph = nx.DiGraph()
            for p in product:
                transitions = [
                    tuple(sorted((*a, b)))
                    for a, b in [*itertools.product([p], cl_names)]
                ]
                for edge in itertools.permutations(transitions, 2):
                    if track_changes:
                        print(edge)
                    graph.add_edge(*edge, weight=background_weight)
            transitions_graphs.update({layer_name: graph})

        # save created graphs as attribute of the object
        self.graph = transitions_graphs
        self.background_weight = background_weight

    def set_transition_canonical(
        self, layer: str, transition: nx.Graph.edges.__class__, weight: float
    ) -> None:
        """
        Set weight of certain transition in propagation model.

        :param layer: name of the later in model
        :param transition: name of transition to be activated, edge in
            propagation model graph
        :param weight: in range (number [0, 1]) of activation
        """
        assert 1 >= weight >= 0, "Weight value should be in [0, 1] range"
        self.graph[layer].edges[transition]["weight"] = weight

    def set_transition_fast(
        self,
        initial_layer_attribute: str,
        final_layer_attribute: str,
        constraint_attributes: Tuple[str, ...],
        weight: float,
    ) -> None:
        """
        Set weight of certain transition in propagation model.

        :param initial_layer_attribute: value of initial attribute which is
            being transited
        :param final_layer_attribute: value of final attribute which is
            being transition
        :param constraint_attributes: other attributes available in the
            propagation model
        :param weight: weight (in range [0, 1]) of activation
        """
        # check data validate
        assert 1 >= weight >= 0, "Weight value should be in [0, 1] range"
        layer_name = initial_layer_attribute.split(".")[0]
        assert (
            layer_name == final_layer_attribute.split(".")[0]
        ), "Layers must have one name!"

        # make transitions
        initial_state = tuple(
            sorted([initial_layer_attribute, *constraint_attributes])
        )
        final_state = tuple(
            sorted([final_layer_attribute, *constraint_attributes])
        )
        self.graph[layer_name].edges[(initial_state, final_state)][
            "weight"
        ] = weight

    def set_transitions_in_random_edges(
        self, weights: List[List[float]]
    ) -> None:
        """
        Set out random transitions in propagation model using given weights.

        :param weights: list of weights to be set in random nodes e.g.
            for model of 3 layers that list [[0.1, 0.2], [0.03, 0.45], [0.55]]
            will change 2 weights in first layer, 2, i second and 1 in third
        """
        # check if given probabilities list is the same as number of layers
        assert len(weights) == len(self.graph), (
            "Len probabilities list and number of layers in model "
            "should be the same!"
        )

        # main loop
        for (name, graph), weight in zip(self.graph.items(), weights):

            edges_list = []
            for wght in weight:

                # select random edge and check if it has not been selected
                # previously
                edge = choice([*graph.edges()])
                while edge in edges_list:
                    edge = choice([*graph.edges()])
                edges_list.append(edge)

                # assign weight to picked edge
                self.set_transition_canonical(name, edge, wght)

    def get_possible_transitions(
        self, state: Tuple[str, ...], layer: str
    ) -> Dict[str, float]:
        """
        Return possible transitions from given state in given layer of model.

        Note that possible transition is a transition with weight > 0.

        :param state: state of the propagation model,
            i.e. ('awareness.UA', 'illness.I', 'vaccination.V')
        :param layer: name of the layer of propagation model from which
            possible transitions are being returned
        :return: dict with possible transitions in shape of:
                 {possible different state in given layer: weight}
        """
        # check if model has been compiled
        assert (
            self.__dict__["graph"] is not None
        ), "Failed to process. Compile model first!"

        # initialise empty container for possible transitions
        states = {}

        # if given general node state doesn't exist in model definition,
        # return all possible states in the given layer with background_weight
        # as possible transitions
        if state not in self.graph[layer]:
            _ = self.get_model_hyperparams()[layer]
            for s_name in _:
                states[s_name] = self.background_weight
            return states

        # otherwise read possible states from model
        for neighbour in self.graph[layer].neighbors(state):
            if self.graph[layer].edges[(state, neighbour)]["weight"] > 0:
                # parse general state to keep only state name in given layer
                for n in neighbour:
                    if layer in n:
                        states[n.split(".")[1]] = self.graph[layer].edges[
                            (state, neighbour)
                        ]["weight"]
        return states
