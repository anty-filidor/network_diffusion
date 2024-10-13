"""Loader of the l2_course_net network."""

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.tpn.tpnetwork import TemporalNetwork


def get_l2_course_net(
    node_features: bool, edge_features: bool, directed: bool
) -> TemporalNetwork:
    """
    Read `l2_course_net`.

    A network built from interactions between students during the three months
    long, abroad language course of the Arabic untertaken by 41 US students.
    This funciton loads a `TemporalNetwork` with three snapshots, each with two
    layers: `"ego"` (with edges picked as top 5 peers by each student) and
    `"course"` (with edges obtained by filtering out a complete graph according
    to a monthly questionarre).

    For details see a paper which analyses the network (constructed slightly
    different - not as a multilayer graph): "Peer Interaction Dynamics and
    Second Language Learning Trajectories During Study Abroad: A Longitudinal
    Investigation Using Dynamic Computational Social Network Analysis" by
    M.B. Paradowski, N. Whitby, M. Czuba, and P. Bródka in "Language Learning",
    2024 (https://doi.org/10.1111/lang.12681).

    :param node_features: wether to to load features of the nodes (mainly
        language profficiency and psychological metrics with personal data).
    :param edge_features: wether to load features of the edges (intensity of
        contact and usage of the second language)
    :param directed: wether to load the network as a directed graph; if so
        nummerical attributes of edges between corresponding nodes are averaged
        while other attributes are discarded

    :return: a temporal, multilayer network
    """
    s_path = Path("/Users/michal/Development/sna-arabic-course/networks")
    snap_1 = _read_snapshot(s_path, 1, node_features, edge_features, directed)
    snap_2 = _read_snapshot(s_path, 2, node_features, edge_features, directed)
    snap_3 = _read_snapshot(s_path, 3, node_features, edge_features, directed)
    # TODO: change the path, remove students' names!
    return TemporalNetwork([snap_1, snap_2, snap_3])


def _read_ego_edges(csv_path: Path, directed: bool) -> pd.DataFrame:
    """
    Read and process edges from the ego layer.

    If network shall not be directed average edge weigths.
    """
    edf = pd.read_csv(csv_path, index_col=0)
    if not directed:
        edf["min_node"] = edf[["source", "target"]].min(axis=1)
        edf["max_node"] = edf[["source", "target"]].max(axis=1)
        edf = (
            edf.groupby(["min_node", "max_node"])
            .agg({"weight": "mean"})
            .reset_index()
        )
        edf = edf.rename(columns={"min_node": "source", "max_node": "target"})
    return edf


def _read_course_edges(csv_path: Path, directed: bool) -> pd.DataFrame:
    """
    Read and process edges from the course layer.

    Filter out edges with too small intensity, normalise numerical colums. If
    the network shall not be directed average numerical columns.
    """
    cdf = pd.read_csv(csv_path, index_col=0)
    cdf = cdf.loc[(cdf["lang_usage"] >= 2.5)]
    cdf["intensity"] = cdf["intensity"] / 5
    cdf["lang_usage"] = cdf["lang_usage"] / 10
    if not directed:
        cdf["min_node"] = cdf[["source", "target"]].min(axis=1)
        cdf["max_node"] = cdf[["source", "target"]].max(axis=1)
        cdf = (
            cdf.groupby(["min_node", "max_node"])
            .agg({"intensity": "mean", "lang_usage": "mean"})
            .reset_index()
        )
        cdf = cdf.rename(columns={"min_node": "source", "max_node": "target"})
    return cdf


def _read_node_attrs(csv_path: Path) -> dict[int, dict[str, Any]]:
    """Read node attributes for a given csv path."""
    df_raw = pd.read_csv(csv_path, index_col=0, sep=",")
    df_final = df_raw.set_index("node_id").T
    return df_final.to_dict()


def _read_snapshot(
    src_path: Path,
    snapshot_idx: int,
    node_features: bool,
    edge_features: bool,
    directed: bool,
) -> MultilayerNetwork:
    """Read given snaphsot of the l2_course network."""
    net_model = nx.DiGraph if directed else nx.Graph

    # read ego layer
    eedges_df = _read_ego_edges(
        src_path / f"ego_edges/{snapshot_idx}_ego_edges.csv", directed
    )
    enet_attrs = list(set(eedges_df.columns).difference({"source", "target"}))
    enet = nx.from_pandas_edgelist(
        eedges_df,
        create_using=net_model,
        edge_attr=enet_attrs if edge_features else None,
    )

    # read course layer
    cedges_df = _read_course_edges(
        src_path / f"course_edges/{snapshot_idx}_course_edges.csv", directed
    )
    cnet_attrs = list(set(cedges_df.columns).difference({"source", "target"}))
    cnet = nx.from_pandas_edgelist(
        cedges_df,
        create_using=net_model,
        edge_attr=cnet_attrs if edge_features else None,
    )

    # read nodes and their features
    if node_features:
        nodes_attrs = _read_node_attrs(src_path / f"{snapshot_idx}_nodes.csv")
        nx.set_node_attributes(cnet, nodes_attrs.copy())
        nx.set_node_attributes(enet, nodes_attrs.copy())

    # couple the layers and return the network
    net = MultilayerNetwork(layers={"ego": enet, "course": cnet})
    return net


if __name__ == "__main__":
    from network_diffusion.mln.functions import draw_mln

    l2_course_net = get_l2_course_net(
        node_features=False, edge_features=False, directed=False
    )
    for snap in l2_course_net.snaps:
        print(snap)
        draw_mln(snap)
