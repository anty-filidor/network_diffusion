from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.nets import l2_course_net

EXP_NODE_ATTRS = {
    "202conv",
    "202_final",
    "202sqr",
    "avgOPI202sq",
    "avgOPI_202",
    "cicle_others_TL",
    "cir_f2fminday",
    "cir_f2fpropL1",
    "cir_f2fpropL2",
    "cir_f2fpropTL",
    "cir_textingminday",
    "cir_textingpropL1",
    "cir_textingpropL2",
    "cir_textingpropTL",
    "cir_voiceminday",
    "cir_voicepropL1",
    "cir_voicepropL2",
    "cir_voicepropTL",
    "cir_writingminday",
    "cir_writingpropL1",
    "cir_writingpropL2",
    "cir_writingpropTL",
    "circle_biolfamily_L1",
    "circle_biolfamily_L2",
    "circle_biolfamily_TL",
    "circle_classmates_L1",
    "circle_classmates_L2",
    "circle_classmates_TL",
    "circle_friends_L1",
    "circle_friends_L2",
    "circle_friends_TL",
    "circle_others_L1",
    "circle_others_L2",
    "circle_partnerwithfamily_L1",
    "circle_partnerwithfamily_L2",
    "circle_partnerwithfamily_TL",
    "circle_teacher_L1",
    "circle_teacher_L2",
    "circle_teacher_TL",
    "context_#daysinTLcountry",
    "context_TLenvironment_stay",
    "context_arrival_TLcountry",
    "context_learningoutofclassminday",
    "context_learntbeforefor#months",
    "context_motivation",
    "group_id",
    "group_late completion",
    "group_questionnaire_order",
    "improv_cult",
    "improv_gmr",
    "improv_listening",
    "improv_overall",
    "improv_pron",
    "improv_reading",
    "improv_voc",
    "improv_writing",
    "interact_classmates",
    "interact_frequency",
    "interaction_groupintegration",
    "living_TLhostfam",
    "living_accommodation",
    "living_communication_L1",
    "living_communication_L2",
    "living_communication_TL",
    "living_flatsharingwith#",
    "living_nonTLhostfam",
    "living_other",
    "living_privatermdorm",
    "living_sharing#L2",
    "living_sharing#TL",
    "living_sharing#sameL1",
    "living_sharingdormroomw#",
    "living_sharingwpartner",
    "living_singleapt",
    "living_sum_flatmates",
    "metric_2nd_L1",
    "metric_Age",
    "metric_English",
    "metric_FLcumulative_wo_AR",
    "metric_FLcumulativecompetence",
    "metric_Gender",
    "metric_L1",
    "metric_eldersiblings",
    "metric_general_cumulativecompet",
    "metric_language@home",
    "metric_level gained",
    "metric_multilingual",
    "metric_postsojournOPI",
    "metric_presojournOPI",
    "metric_sameagedsiblings",
    "metric_sum_FL",
    "metric_sum_languagesfromTLgroup",
    "metric_sum_siblings",
    "metric_targetlanguage",
    "metric_youngersiblings",
    "partner_01",
    "partner_language",
    "partner_stayingwith",
    "partner_talkevery#days",
    "prog202conv",
    "prog202sqr",
    "progOPI202sqr",
    "progOPI_202",
    "psycho_TLcomfort",
    "psycho_TLcourseenjoyment",
    "psycho_TLmistakes",
    "psycho_TLstudyenjoyment",
    "psycho_TLuseoutofclassminday",
    "psycho_attention",
    "psycho_avoidance_BAL2",
    "psycho_fear",
    "psycho_friends_BAL2",
    "psycho_initiative_BAL2",
    "psycho_learninglanguage_BAL2",
    "psycho_listener_BAL2",
    "psycho_mix_BAL2",
    "psycho_motivationdegree",
    "psycho_otherlgsminday",
    "psycho_proficiencyingroup_BAL1",
    "psycho_studytogether",
    "qasid_class",
    "status",
}


EXP_EDGE_ATTRS = {
    "ego": {"weight"},
    "course": {"direction", "intensity", "lang_usage"},
}


@pytest.mark.parametrize(
    "snap_idx, directed, exp_l_names, exp_nb_actors, exp_nb_nodes, exp_nb_edges, exp_avg_deg, exp_cc",
    [
        (
            1,
            False,
            {"ego", "course"},
            41,
            [41, 41],
            [116, 181],
            [5.659, 8.829],
            [0.485, 0.564],
        ),
        (
            2,
            False,
            {"ego", "course"},
            41,
            [41, 41],
            [126, 169],
            [6.146, 8.244],
            [0.398, 0.563],
        ),
        (
            3,
            False,
            {"ego", "course"},
            41,
            [41, 41],
            [114, 152],
            [5.561, 7.415],
            [0.402, 0.651],
        ),
        (
            1,
            True,
            {"ego", "course"},
            41,
            [41, 41],
            [175, 228],
            [8.537, 11.122],
            [0.443, 0.404],
        ),
        (
            2,
            True,
            {"ego", "course"},
            41,
            [41, 41],
            [185, 209],
            [9.024, 10.195],
            [0.346, 0.383],
        ),
        (
            3,
            True,
            {"ego", "course"},
            41,
            [41, 41],
            [159, 177],
            [7.756, 8.634],
            [0.330, 0.418],
        ),
    ],
)
def test_read_snapshot_undirected(
    snap_idx,
    directed,
    exp_l_names,
    exp_nb_actors,
    exp_nb_nodes,
    exp_nb_edges,
    exp_avg_deg,
    exp_cc,
):
    snap_net: MultilayerNetwork = l2_course_net._read_snapshot(
        src_path=Path(__file__).parent.parent.parent / "nets/data",
        s_idx=snap_idx,
        node_features=False,
        edge_features=False,
        directed=directed,
    )
    assert set(snap_net.get_layer_names()) == exp_l_names
    assert snap_net.get_actors_num() == exp_nb_actors
    for el_n, el_e, el_ad, el_cc, l_name in zip(
        exp_nb_nodes, exp_nb_edges, exp_avg_deg, exp_cc, ["ego", "course"]
    ):
        l_graph = snap_net[l_name]
        assert len(l_graph.nodes()) == el_n
        assert len(l_graph.edges()) == el_e
        assert round(np.average([d for _, d in l_graph.degree]), 3) == el_ad
        assert round(nx.average_clustering(l_graph), 3) == el_cc


def test_node_features():
    net = l2_course_net.get_l2_course_net(
        node_features=True,
        edge_features=False,
        directed=False,
    )
    for s_idx, snap in enumerate(net.snaps):
        for l_name, l_graph in snap.layers.items():
            for node, attrs in list(l_graph.nodes(data=True)):
                assert (
                    set(attrs.keys()) == EXP_NODE_ATTRS
                ), f"error in snap {s_idx}, layer {l_name}, node {node}"


def test_edge_features():
    net = l2_course_net.get_l2_course_net(
        node_features=False,
        edge_features=True,
        directed=True,
    )
    for s_idx, snap in enumerate(net.snaps):
        for l_name in snap.get_layer_names():
            for n_a, n_b, attrs in list(snap[l_name].edges(data=True)):
                assert (
                    set(attrs.keys()) == EXP_EDGE_ATTRS[l_name]
                ), f"error in snap {s_idx}, layer {l_name}, edge {n_a}, {n_b}"
