# Copyright 2023 by D. Dąbrowski, M. Czuba, P. Bródka. All Rights Reserved.
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

"""A definition community based influence maximization selector class."""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.seeding.base_selector import BaseSeedSelector
from network_diffusion.seeding.degreecentrality_selector import (
    DegreeCentralitySelector,
)
from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class ListsEws:
    """Valueles for lists in the CBIMselector.Ews function."""

    actor_list: List[MLNetworkActor] = []
    sort_list: List[MLNetworkActor] = []
    ews_actor: Dict[MLNetworkActor, int] = {}
    degree_actor: Dict[MLNetworkActor, int] = {}
    seed_size: float = 0


class CBIMselector(BaseSeedSelector):
    """
    CBIM seed selector.

    This is an implementation based on:
    Chen, X., Deng, L., Zhao, Y. et al.
    Community-based influence maximization in location-based social network.
    World Wide Web 24, 1903–1928 (2021).
    https://doi.org/10.1007/s11280-021-00935-x
    """

    def __init__(
        self, threshold: float, seed_size: Optional[float] = None
    ) -> None:
        """
        Create an object.

        :param threshold: a threshold proper to the CBIM.
        :param seed_size: seed size to compute a ranking for; if not provided
            then selector works on the bse parameters and includes all
            parameters in the calculatopns.
        """
        super().__init__()
        assert (
            seed_size is None or 0 <= seed_size < 1
        ), f"incorrect seed_size value: {seed_size}!"
        if seed_size is None:
            self.seed_default = True  # seed_default say if seed_size is None
        else:
            self.seed_default = False
            ListsEws.seed_size = seed_size
        self.threshold = threshold

    alpha = 0.1  # can be different, but in article that is the best

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tCBIM choice\n{BOLD_UNDERLINE}\n"
        )

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking using Community Based Influence Maximization."""
        lis: set[MLNetworkActor] = set()
        actors_l = self.cbim(net)

        size_l = ListsEws.seed_size // len(
            net.layers
        )  # how many nodes to select from one layer
        size_r = ListsEws.seed_size % len(
            net.layers
        )  # the rest nodes to select which are not in size_l
        lis = self.ranking_list(
            net=net, actors_l=actors_l, size_l=size_l, lis=lis
        )
        lis = lis.union(
            self.ranking_list_additional(net, size_r, actors_l, lis)
        )  # adding chosen nodes to lis from the rest size

        ranking_list = list(lis)
        for (
            actor
        ) in net.get_actors():  # checking if each actor has added to list
            if actor not in ranking_list:
                ranking_list.append(actor)

        return ranking_list

    def cbim(
        self,
        net: MultilayerNetwork,
    ) -> Dict[str, set[MLNetworkActor]]:
        """
        Calculate cbim.

        :param net: Multilayer Network
        :return: actors with calculated ews for each node
        """
        communities = pd.DataFrame(self.create_by_degree(net))

        for l_name in net.layers:
            # selecting seed size by procent of whole
            for (
                community,
                _,
            ) in communities.iterrows():  # calculate merge for each community
                communities.at[community, f"Merge{l_name}"] = self.merge_index(
                    net, communities, str(community), l_name
                )
        merged_communities = self.merge_communities(
            net, communities
        )  # merge communities

        actors = self.seed_size_to_calculate_ews(net, merged_communities)

        return actors

    @staticmethod
    def create_by_degree(
        net: MultilayerNetwork,
    ) -> Dict[str, Dict[str, List[MLNetworkActor]]]:
        """
        :param net: Multilayer Network.

        :return: dict with information about layer, community and actor.
        """
        community: Dict[
            str, Dict[str, List[MLNetworkActor]]
        ] = {}  # crating group: name_layer, group, actor
        com_list: List[
            MLNetworkActor
        ] = []  # a list with each node which is in some community
        degre = list(DegreeCentralitySelector().actorwise(net))

        for l_name in net.layers:
            com_list.clear()  # after change layer, the list is cleared
            grap: nx.Graph = net.layers[l_name]
            i = 0  # number of community for the next layer is cleared
            # creating groups
            while len(com_list) < len(net.layers[l_name]):
                for actor in degre:
                    if (
                        l_name not in actor.layers
                    ):  # the condition for the aucs network
                        continue
                    if actor in com_list:  # actor can't be in com_list
                        continue
                    group = f"C{i}"

                    i += 1
                    if community.get(l_name) is None:  # initialization dict
                        community[l_name] = {}
                        community[l_name][group] = []
                    if (
                        community[l_name].get(group) is None
                    ):  # initialization list in dict
                        community[l_name][group] = []
                        community[l_name][group].append(
                            actor
                        )  # adding actor to community
                        com_list.append(actor)  # adding to global list
                    for neighbour in grap.neighbors(actor.actor_id):
                        if net.get_actor(neighbour) in com_list:
                            continue

                        com_list.append(net.get_actor(neighbour))
                        community[l_name][group].append(
                            net.get_actor(neighbour)
                        )

        return community

    @staticmethod
    def merge_index(
        net: MultilayerNetwork, cbim: pd.DataFrame, community: str, l_name: str
    ) -> float:
        """
        Calculate merge index for community.

        :param net: a multilayer network
        :param cbim: database in pandas saving information about communities
        :param community: selected community to calculate merge index
        :param l_name: name of layer
        """
        graph: nx.Graph = net.layers[l_name]
        ein, eout = 0, 0

        actors = cbim.at[community, f"{l_name}"]
        if actors is np.nan:
            return 0
        for _, actor in enumerate(actors):
            for neighbour in graph.neighbors(actor.agent_id):

                if net.get_actor(neighbour) in actors:
                    ein = ein + 1
                else:
                    eout = eout + 1
        if (
            eout == 0 and ein == 0
        ):  # conditions for node which don't have neighbours in the layer
            return 0
        community_scale = len(actors) / len(graph.nodes)
        community_conductance = eout / (2 * ein + eout)
        merging_index = community_scale * community_conductance

        return merging_index

    def merge_communities(
        self, net: MultilayerNetwork, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge pairs of weak communities into a one.

        :param net: Multilayer Network
        :param data: database in pandas which has information about communities
        :param threshold: the procent of average maximum values of merging idx
            from each layer
        :return: a dataframe with communities
        """
        assert (
            0 <= self.threshold <= 1
        ), f"incorrect threshold: {self.threshold}!"
        tab: Dict[str, float] = {}
        new_community: pd.DataFrame  # it can be new community
        merging_index_threshold = (
            self.threshold
            * sum(max(data[f"Merge{layer}"]) for layer in net.layers)
            / len(net.layers)
        )  # threshold to compare values

        for l_name in net.layers:

            tab.clear()
            minimal = (
                data[f"Merge{l_name}"].loc[lambda x: x != 0].min()
            )  # condition for minimal value
            minimum = data.loc[
                data[f"Merge{l_name}"] == minimal
            ]  # select minimal value

            while merging_index_threshold > sum(
                # merging communities to merging_index_threshold,
                minimum[f"Merge{l_name}"]
            ) / len(minimum[f"Merge{l_name}"]):
                for (
                    idx,
                    val,
                ) in minimum.iterrows():  # communities with min value

                    tab.clear()
                    if (  # conditions if values don't exist
                        val[f"{l_name}"] != data.loc[idx, f"{l_name}"]
                        or val[f"{l_name}"] is np.nan
                        or len(val[f"{l_name}"]) == 0
                    ):
                        continue
                    for (
                        idx2,
                        val2,
                    ) in data.iterrows():  # second communities to compare

                        if (
                            val2[f"{l_name}"] is np.nan or idx == idx2
                        ):  # conditions values can't be none and be the same
                            continue
                        if (
                            val2[f"{l_name}"] == 0
                            or len(val2[f"{l_name}"]) == 0
                        ):  # a value in row can't be 0 or empty
                            continue
                        tab.update(  # dict with communities and value
                            self.dice_similarity_between_nodes(
                                str(idx2),
                                val[f"{l_name}"],
                                val2[f"{l_name}"],
                                net,
                                l_name,
                            )
                        )

                    community_to_merge = max(tab.items(), key=lambda x: x[1])[
                        0
                    ]  # select maks value

                    # it collects communities which are free or cleared
                    new_community = data.loc[
                        data[f"{l_name}"].apply(
                            lambda x: x == 0 or x is np.nan
                        ),
                        f"{l_name}",
                    ]

                    while len(new_community.index.to_list()) == 0:

                        data.at[
                            f"C{len(data.index)}", f"{l_name}"
                        ] = 0  # adding value and creating new index

                        new_community = (
                            data.loc[  # a new which can be new place to save
                                data[f"{l_name}"].apply(
                                    lambda x: x == 0 or len(x) == 0
                                ),
                                f"{l_name}",
                            ]
                        )
                    data, minimum = self.communities_change(
                        net,
                        data,
                        new_community,
                        [community_to_merge, l_name, idx],
                    )

                if len(minimum) == 0:
                    break

        return data

    @staticmethod
    def dice_similarity_between_nodes(
        c: str,
        node1: List[MLNetworkActor],
        node2: List[MLNetworkActor],
        net: MultilayerNetwork,
        l_name: str,
    ) -> Dict[str, float]:
        """
        Calculate dice similarity between nodes in communities.

        :param c: name second community to calculate
        :param node1: list of actors in first community
        :param node2: lis of actors in second community
        :param net: Multilayer Network
        :param l_name: name of layer
        """
        graph: nx.Graph = net.layers[l_name]
        dsc = 0.0
        similarities: Dict[str, float] = {}

        for _, actor in enumerate(node1):  # actors from first community
            node1_len = set(
                graph.neighbors(actor.actor_id)
            )  # neighbours node 1
            for _, actor2 in enumerate(node2):  # actors from second community

                node2_len = set(
                    graph.neighbors(actor2.actor_id)
                )  # neighbours node 2
                if node2_len == 0:  # list of neighbour can not be empty
                    continue
                neighbors_inter = len(
                    node1_len.intersection(node2_len)
                )  # nodes shared by two communities

                union = len(node1_len) + len(
                    node2_len
                )  # sum nodes od two communities
                if union == 0:  # union can't be equal 0
                    similarities[c] = 0
                    return similarities
                dsc = dsc + (2 * neighbors_inter / union)

        similarities[c] = dsc / (len(node2))
        return similarities

    def ews(
        self, net: MultilayerNetwork, cbim: pd.DataFrame, l_name: str
    ) -> set:
        """
        Calculate parametr edge sum weight.

        :param net: Multilayer Network
        :param cbim: database in pandas which has information about communities
        :param l_name: name of layer

        :return: start the next function
        """
        graph: nx.Graph = net.layers[l_name]
        cbim[f"Quota{l_name}"] = 0
        cbim[f"Sorted{l_name}"] = None
        cbim[f"Sorted{l_name}"] = cbim[f"Sorted{l_name}"].astype(object)
        for (
            col,
            row,
        ) in cbim.iterrows():  # enumerate by community number and date

            ListsEws.ews_actor.clear()
            data = row[f"{l_name}"]
            ListsEws.actor_list.clear()

            if (data != 0 and len(str(data)) > 0) and data is not np.nan:
                ListsEws.actor_list = []
                ListsEws.degree_actor = {}
                self.adj_matrix_calculate(data, graph)

                if len(ListsEws.sort_list) != 0:
                    cbim.at[
                        col, f"Sorted{l_name}"
                    ] = (
                        ListsEws.sort_list.copy()
                    )  # saving sorted actors in Sorted label

        return self.quota(cbim, net, l_name)

    @staticmethod
    def quota(cbim: pd.DataFrame, net: MultilayerNetwork, l_name: str) -> set:
        """
        Select actor to take from each community in layer by defined number.

        :param cbim: database with information about communities
        :param net: Multilayernetwork
        :param l_name: name of layer

        :return: sorted seed for each layer
        """
        graph: nx.Graph = net.layers[l_name]
        nodes_to_activated = 0
        quota: Dict[str, float] = {}
        seed = set()
        for idx, val in cbim.iterrows():
            if val[f"Sorted{l_name}"] is None:
                continue

            quota_value = ListsEws.seed_size * (
                len(val[f"Sorted{l_name}"]) / len(graph.nodes)
            )
            quota[str(idx)] = quota_value
            if (
                nodes_to_activated == ListsEws.seed_size
            ):  # this is limited, big size of seed can inflate the result
                break
            for idx1, actor in enumerate(val[f"Sorted{l_name}"]):
                if idx1 == round(quota_value):
                    break
                seed.add(actor)
                nodes_to_activated += 1

        # for those elements which weren't selected and budget is higher
        while (
            nodes_to_activated < ListsEws.seed_size
        ):  # selected nodes must be equal seed_size
            for comm, _ in sorted(quota.items(), key=lambda x: x[1]):
                if nodes_to_activated == ListsEws.seed_size:
                    break
                for _, actor in enumerate(cbim.loc[comm, f"Sorted{l_name}"]):
                    if actor in seed:
                        continue
                    seed.add(actor)  # adding actor to set of seed

                    nodes_to_activated += 1
                    break

        return seed

    def seed_size_to_calculate_ews(
        self,
        net: MultilayerNetwork,
        merged_communities: pd.DataFrame,
    ) -> Dict[str, set[MLNetworkActor]]:
        """
        Calculatee seed_size and :parametr ews.

        :param net: load Multilayer Network
        :param merged_communities: database after merging

        :return: actors with ews :parametr
        """
        actors: Dict[str, set[MLNetworkActor]] = {}
        for l_name in net.layers:
            if self.seed_default:  # select seed_size again for calculate ews
                ListsEws.seed_size = len(net.layers[l_name])
            else:
                ListsEws.seed_size = ListsEws.seed_size * len(
                    net.layers[l_name]
                )
            actors[l_name] = self.ews(net, merged_communities, l_name)
        return actors

    @staticmethod
    def ranking_list(
        net: MultilayerNetwork,
        actors_l: Dict[str, set[MLNetworkActor]],
        size_l: float,
        lis: set[MLNetworkActor],
    ) -> set[MLNetworkActor]:
        """
        Select actors where actors are taken equal from each layer.

        :param net: load Multilayer Network
        :param actors_l: actors sorted descending to the best to take
        :param size_l: number actors to take from each layer
        :param lis: selected actors

        :return: set of selected actors
        """
        for l_name in net.layers:
            size = 0
            actors = [
                values for _, values in enumerate(actors_l[l_name])
            ]  # actors from select layer
            while size < size_l:

                if len(actors) == 0:
                    break
                while actors[0] in lis:
                    del actors[0]  # if actor is in lis remove it
                lis.add(actors[0])
                # remove actor help checking list if it has already selected
                del actors[0]
                size = size + 1
            actors_l[l_name] = set(actors)
        return lis

    @staticmethod
    def ranking_list_additional(
        net: MultilayerNetwork,
        size_r: float,
        actors_l: Dict[str, set[MLNetworkActor]],
        lis: set[MLNetworkActor],
    ) -> set[MLNetworkActor]:
        """
        Select the others node form size_r.

        :param net: load Multilayer Network
        :param size_r: number of actors to suplement
        :param actors_l: actors which are sorted to take
        lis: set of actors to select

        :return: selected actors
        """
        size = 0
        for (
            l_name
        ) in net.layers:  # loop for actors which are not included in size_l
            if size == size_r:
                break

            actors = [values for _, values in enumerate(actors_l[l_name])]
            if len(actors) == 0:
                continue
            while actors[0] in lis:  # checking if actor has already selected
                del actors[0]
            lis.add(actors[0])  # adding to list
            del actors[0]
            size = size + 1
            continue
        return lis

    def communities_change(
        self,
        net: MultilayerNetwork,
        data: pd.DataFrame,
        new_community: pd.DataFrame,
        community_info: List,  # community to merge , l_name, idx
    ) -> Tuple[pd.DataFrame, float]:
        """
        Create new community, clear the olds and searching the new minimal.

        :param net: load Multilayer Network
        :param data: database with communities attributes
        :param new_community: new place to save merged community
        :param community_infor: a list which include information
        about community to merge,
         layer name and first community index to merge

        :return: tuple with database and minimal value
        """
        community_candydat = new_community.index.to_list()[
            0
        ]  # selected community to save information

        data.at[community_candydat, f"{community_info[1]}"] = (
            data.loc[community_info[0], f"{community_info[1]}"]
            + data.loc[
                community_info[2], f"{community_info[1]}"
            ]  # merging communities
        )
        data.at[
            community_candydat, f"Merge{community_info[1]}"
        ] = self.merge_index(
            net, data, community_candydat, community_info[1]
        )  # calculate merge index for new community
        data.at[
            community_info[0], f"{community_info[1]}"
        ] = np.nan  # clear value and create empty place for new community
        data.at[community_info[0], f"Merge{community_info[1]}"] = 0
        data.at[
            community_info[2], f"{community_info[1]}"
        ] = np.nan  # clear value and create empty place for new community
        data.at[community_info[2], f"Merge{community_info[1]}"] = 0
        minimal = data[f"Merge{community_info[1]}"].loc[lambda x: x != 0].min()
        minimum = data.loc[
            data[f"Merge{community_info[1]}"]
            == minimal  # get minimum value from Merge label
        ]
        return data, minimum

    def adj_matrix_calculate(
        self, data: pd.DataFrame, graph: nx.Graph
    ) -> None:
        """
        Calculate adjacency matrix and ews paramentr.

        :param data: database with communities attributes
        :param graph: generated graph from layer
        """
        for _, val in enumerate(data):  # enumerate actors to calculate ews
            ListsEws.degree_actor[val.agent_id] = graph.degree[val.agent_id]
            ListsEws.actor_list.append(val.agent_id)

        adj_matrix = nx.adjacency_matrix(
            graph, ListsEws.actor_list
        ).toarray()  # adjacency matrix to type array

        for _, val in enumerate(data):

            ews = 0  # ews for node in community

            for length in range(1, len(graph.nodes) + 1):
                ews += np.sum(
                    pow(self.alpha, length)
                    * ListsEws.degree_actor[val.agent_id]
                    * np.linalg.matrix_power(adj_matrix, length)
                )
            ListsEws.ews_actor[val] = ews
        ListsEws.sort_list.clear()
        if len(ListsEws.ews_actor.keys()) == len(ListsEws.ews_actor.values()):
            # Convert dictionary to DataFrame
            dataf = pd.DataFrame(  # create it like dataframe
                ListsEws.ews_actor.items(), columns=["Key", "Value"]
            )
            ListsEws.sort_list = dataf.sort_values(
                by="Value", ascending=False
            )["Key"].tolist()

        else:
            raise ValueError("Error: Length of keys and values must be equal.")
