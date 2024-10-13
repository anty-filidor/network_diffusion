from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.tpn.tpnetwork import TemporalNetwork

src_path = Path("/Users/michal/Development/sna-arabic-course/networks")


def get_l2_course_net() -> TemporalNetwork:
    snap_1 = _read_snapshot(1)
    snap_2 = _read_snapshot(2)
    snap_3 = _read_snapshot(3)
    return TemporalNetwork([snap_1, snap_2, snap_3])


def _read_attrs(csv_path: Path) -> dict[int, dict[str, Any]]:
    df_raw = pd.read_csv(csv_path, index_col=0, sep=",")
    df_final = df_raw.set_index("node_id").T
    return df_final.to_dict()


def _read_snapshot(snapshot_idx: int) -> MultilayerNetwork:
    """Read one snaphsot of the network."""

    # read nodes and their features
    nodes_attrs = _read_attrs(src_path / f"{snapshot_idx}_nodes.csv")

    # read ego edges which are properly prepcosessed and normalised
    eedges_path = src_path / f"ego_edges/{snapshot_idx}_ego_edges.csv"
    eedges_df = pd.read_csv(eedges_path, index_col=0)
    enet = nx.from_pandas_edgelist(
        eedges_df,
        create_using=nx.DiGraph,
        edge_attr=["weight"],
    )
    nx.set_node_attributes(enet, nodes_attrs.copy())

    # read course edges, filter out those with small intensity and normalise
    cedges_path = src_path / f"course_edges/{snapshot_idx}_course_edges.csv"
    cedges_df = pd.read_csv(cedges_path, index_col=0)
    cedges_df = cedges_df.loc[(cedges_df["lang_usage"] >= 2.5)]
    cedges_df["intensity"] = cedges_df["intensity"] / 5
    cedges_df["lang_usage"] = cedges_df["lang_usage"] / 10
    cnet = nx.from_pandas_edgelist(
        cedges_df,
        create_using=nx.DiGraph,
        edge_attr=["direction", "intensity", "lang_usage"],
    )
    nx.set_node_attributes(cnet, nodes_attrs.copy())

    # couple the laters and return the network
    net = MultilayerNetwork(layers={"ego": enet, "course": cnet})
    print(net)
    return net


if __name__ == "__main__":
    get_l2_course_net()
