"""A definition community based influence maximization selector class."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

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


class CBIMselector(BaseSeedSelector):
    """CBIM seed selector."""

    def __init__(self, threshold: float, seed_size: int = None) -> None:
        """Object."""
        super().__init__()
        assert (
            seed_size is None or 0 <= seed_size <= 100
        ), f"incorrect seed_size value: {seed_size}!"
        if seed_size is None:
            self.seed_default = True  # seed_default say if seed_size is None
        else:
            self.seed_default = False
            self.seed_proc = (
                seed_size / 100
            )  # size like procent of the hole graph
        self.threshold = threshold

    alpha = 0.1  # can be different, but in article that is the best

    def __str__(self) -> str:
        """Return seed method's description."""
        return (
            f"{BOLD_UNDERLINE}\nseed selection method\n{THIN_UNDERLINE}\n"
            f"\tCBIM choice\n{BOLD_UNDERLINE}\n"
        )

    @dataclass
    class ListsEws:
        """Valuables for lists in ews function."""

        actor_list: List[MLNetworkActor] = field(default_factory=list)
        sort_list: List[MLNetworkActor] = field(default_factory=list)
        ews_actor: Dict[MLNetworkActor, int] = field(default_factory=dict)
        degree_actor: Dict[MLNetworkActor, int] = field(default_factory=dict)
        seed_size: float = 0

    @staticmethod
    def _calculate_ranking_list(graph: nx.Graph) -> List[Any]:
        """Create nodewise ranking."""
        raise NotImplementedError(
            "Nodewise ranking list cannot be computed for this class!"
        )

    def actorwise(self, net: MultilayerNetwork) -> List[MLNetworkActor]:
        """Get ranking using Community Based influence maximization."""
        lis: set[MLNetworkActor] = set()
        actors_l = self.cbim(net)

        size_l = self.ListsEws.seed_size // len(
            net.layers
        )  # how many nodes to select from one layer
        size_r = self.ListsEws.seed_size % len(
            net.layers
        )  # the rest to select

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

        ranking_list = list(lis)
        # print(len(net.get_actors()))
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

        param_net: Multilayer Network
        """
        actors: Dict[str, set[MLNetworkActor]] = {}

        communities = pd.DataFrame(self.create_by_degree(net))

        for l_name in net.layers:
            # if self.seed_default:
            #   self.seed_size = len(net.layers[l_name])
            # else:
            # self.seed_size = self.seed_proc * len(
            #   net.layers[l_name]
            # )  # selecting seed size by procent of whole
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

        for l_name in net.layers:
            if self.seed_default:  # select seed_size again for calculate ews
                self.ListsEws.seed_size = len(net.layers[l_name])
            else:
                self.ListsEws.seed_size = self.seed_proc * len(
                    net.layers[l_name]
                )
            actors[l_name] = self.ews(net, merged_communities, l_name)
        return actors

    @staticmethod
    def create_by_degree(
        net: MultilayerNetwork,
    ) -> Dict[str, Dict[str, List[MLNetworkActor]]]:
        """
        param_net: Multilayer Network.

        return: dict with information about layer, community and actor.
        """
        community: Dict[
            str, Dict[str, List[MLNetworkActor]]
        ] = {}  # crating group: name_layer, group, actor
        com_list: List[
            MLNetworkActor
        ] = []  # a list with each node which is in some community
        deg = DegreeCentralitySelector()
        degre = list(deg.actorwise(net))
        # print(len(degre))
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

        param_net: Multilayer Network
        param_cbim: database in pandas saving information about communities
        param_community: selected community to calculate merge index
        param_l_name: name of layer
        """
        graph: nx.Graph = net.layers[l_name]
        ein, eout = 0, 0

        actors = cbim.at[community, f"{l_name}"]
        if actors is np.nan:
            return 0
        for _, actor in enumerate(actors):
            for neighbour in graph.neighbors(actor.actor_id):

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
        Two weak communities change on one .

        param_net: Multilayer Network
        param_data: database in pandas which has information about communities
        param_threshold: the procent of average maximum values of merging index
        from each layer

        """
        assert (
            0 <= self.threshold <= 1
        ), f"incorrect threshold: {self.threshold}!"
        tab: Dict[str, float] = {}
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
                    new_community: pd.DataFrame  # it can be new community

                    # it collects communities which are free or cleared
                    new_community = data.loc[
                        data[f"{l_name}"].apply(
                            lambda x: x == 0 or x is np.nan
                        ),
                        f"{l_name}",
                    ]
                    # print(f"COMUNITY:{new_community}")
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

                    community_candydat = new_community.index.to_list()[
                        0
                    ]  # selected community to save information
                    # print(new_community)

                    data.at[community_candydat, f"{l_name}"] = (
                        data.loc[community_to_merge, f"{l_name}"]
                        + data.loc[idx, f"{l_name}"]  # merging communities
                    )
                    data.at[
                        community_candydat, f"Merge{l_name}"
                    ] = self.merge_index(
                        net, data, community_candydat, l_name
                    )  # calculate merge index for new community
                    data.at[
                        community_to_merge, f"{l_name}"
                    ] = (
                        np.nan
                    )  # clear value and create empty place for new community
                    data.at[community_to_merge, f"Merge{l_name}"] = 0
                    data.at[
                        idx, f"{l_name}"
                    ] = (
                        np.nan
                    )  # clear value and create empty place for new community
                    data.at[idx, f"Merge{l_name}"] = 0
                    minimal = (
                        data[f"Merge{l_name}"].loc[lambda x: x != 0].min()
                    )
                    minimum = data.loc[
                        data[f"Merge{l_name}"]
                        == minimal  # get minimum value from Merge label
                    ]
                if len(minimum) == 0:
                    break

            # print(data)
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

        param_c: name second community to calculate
        param_node1: list of actors in first community
        param_node2: lis of actors in second community
        param_net: Multilayer Network
        param_l_name: name of layer
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

        param_net: Multilayer Network
        param_cbim: database in pandas which has information about communities
        param_l_name: name of layer
        """
        # actor_ews: Dict[MLNetworkActor, int] = {}
        graph: nx.Graph = net.layers[l_name]
        extra = self.ListsEws()
        cbim[f"Quota{l_name}"] = 0
        cbim[f"Sorted{l_name}"] = None
        cbim[f"Sorted{l_name}"] = cbim[f"Sorted{l_name}"].astype(object)
        for (
            col,
            row,
        ) in cbim.iterrows():  # enumerate by community number and date

            extra.ews_actor.clear()
            # actor_ews.clear()
            data = row[f"{l_name}"]
            extra.actor_list.clear()
            # list_actors.clear()

            if (data != 0 and len(str(data)) > 0) and data is not np.nan:
                extra.actor_list = []
                # list_actors = []
                extra.degree_actor = {}
                # dict_degree = {}
                # print(col)

                for _, val in enumerate(
                    data
                ):  # enumerate actors to calculate ews
                    extra.degree_actor[val.actor_id] = graph.degree[
                        val.actor_id
                    ]
                    # dict_degree[val.actor_id] = graph.degree[val.actor_id]
                    extra.actor_list.append(val.actor_id)

                adj_matrix = nx.adjacency_matrix(
                    graph, extra.actor_list
                ).toarray()  # adjacency matrix to type array

                for _, val in enumerate(data):

                    ews = 0  # ews for node in community

                    for length in range(1, len(graph.nodes) + 1):
                        ews += np.sum(
                            pow(self.alpha, length)
                            * extra.degree_actor[val.actor_id]
                            * np.linalg.matrix_power(adj_matrix, length)
                        )
                    extra.ews_actor[val] = ews
                    # actor_ews[val] = ews
                extra.sort_list.clear()
                # sorted_list.clear()
                if len(extra.ews_actor.keys()) == len(
                    extra.ews_actor.values()
                ):
                    # Convert dictionary to DataFrame
                    dataf = pd.DataFrame(  # create it like dataframe
                        extra.ews_actor.items(), columns=["Key", "Value"]
                    )
                    extra.sort_list = dataf.sort_values(
                        by="Value", ascending=False
                    )["Key"].tolist()

                else:
                    raise ValueError(
                        "Error: Length of keys and values must be equal."
                    )

                if len(extra.sort_list) != 0:
                    cbim.at[
                        col, f"Sorted{l_name}"
                    ] = (
                        extra.sort_list.copy()
                    )  # saving sorted actors in Sorted label
                # print(cbim["Sorted{}".format(l_name)])
        return self.quota(cbim, net, l_name)

    def quota(
        self, cbim: pd.DataFrame, net: MultilayerNetwork, l_name: str
    ) -> set:
        """
        Select actor to take from each community in layer by defined number.

        param_cbim: database with information about communities
        param_net: Multilayernetwork
        param_l_name: name of layer
        """
        graph: nx.Graph = net.layers[l_name]
        nodes_to_activated = 0
        quota: Dict[str, float] = {}
        seed = set()
        for idx, val in cbim.iterrows():
            if val[f"Sorted{l_name}"] is None:
                continue

            quota_value = self.ListsEws.seed_size * (
                len(val[f"Sorted{l_name}"]) / len(graph.nodes)
            )
            quota[str(idx)] = quota_value
            if (
                nodes_to_activated == self.ListsEws.seed_size
            ):  # this is limited, big size of seed can inflate the result
                break
            for idx1, actor in enumerate(val[f"Sorted{l_name}"]):
                if idx1 == round(quota_value):
                    break
                seed.add(actor)
                nodes_to_activated += 1

        # for those elements which weren't selected and budget is higher
        while (
            nodes_to_activated < self.ListsEws.seed_size
        ):  # selected nodes must be equal seed_size
            for comm, _ in sorted(quota.items(), key=lambda x: x[1]):
                if nodes_to_activated == self.ListsEws.seed_size:
                    break
                for _, actor in enumerate(cbim.loc[comm, f"Sorted{l_name}"]):
                    if actor in seed:
                        continue
                    seed.add(actor)  # adding actor to set of seed

                    nodes_to_activated += 1
                    break

        # print(f"seed:{seed}\n len:{len(seed)}")
        return seed
