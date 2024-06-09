# flake8: noqa
# pylint: disable-all
# isort: skip_file

# 1st setcion

import networkx as nx
import network_diffusion as nd
import numpy as np

from network_diffusion.models import (
    BaseModel,
    CompartmentalGraph,
    NetworkUpdateBuffer,
)

# 2nd section


class SIR_UAModel(BaseModel):

    def __init__(
        self,
        alpha: float,
        alpha_prime: float,
        beta: float,
        gamma: float,
        delta: float,
        ill_seeds: float,
        aware_seeds: float,
    ) -> None:
        """
        A model SIR~UA.

        :param alpha: weight prob. of S->I for unaware nodes
        :param alpha_prime: prob. of S->I for aware nodes
        :param beta: prob. of I->R for both unaware and aware nodes
        :param gamma: prob. of U->A for both suspected and removed nodes
        :param delta: prob. of U->A for ill nodes
        :param ill_seeds: % of initially I nodes
        :param aware_seeds: % of initially A nodes
        """
        compartments = self._create_compartments(
            alpha=alpha,
            alpha_prime=alpha_prime,
            beta=beta,
            gamma=gamma,
            delta=delta,
            ill_seeds=ill_seeds,
            aware_seeds=aware_seeds,
        )
        self.__comp_graph = compartments
        self.__seed_selector = nd.seeding.RandomSeedSelector()

    @staticmethod
    def _create_compartments(
        alpha: float,
        alpha_prime: float,
        beta: float,
        gamma: float,
        delta: float,
        ill_seeds: float,
        aware_seeds: float,
    ) -> CompartmentalGraph:
        # define processes, allowed states and initial % of actors in that states
        phenomena: dict[str, tuple] = {
            "contagion": (["S", "I", "R"], [100 - ill_seeds, ill_seeds, 0]),
            "awareness": (["U", "A"], [100 - aware_seeds, aware_seeds]),
        }

        # wrap them into a compartments
        cg = CompartmentalGraph()
        for phenomenon, (states, budget) in phenomena.items():
            cg.add(process_name=phenomenon, states=states)  # name of process
            cg.seeding_budget.update({phenomenon: budget})  # initial %s
        cg.compile(background_weight=0)

        # set up weights of transitions for SIR and unaware
        cg.set_transition_fast(
            "contagion.S", "contagion.I", ("awareness.U",), alpha
        )
        cg.set_transition_fast(
            "contagion.I", "contagion.R", ("awareness.U",), beta
        )

        # set up weights of transitions for SIR and aware
        cg.set_transition_fast(
            "contagion.S", "contagion.I", ("awareness.A",), alpha_prime
        )
        cg.set_transition_fast(
            "contagion.I", "contagion.R", ("awareness.A",), beta
        )

        # set up weights of transitions for UA and suspected
        cg.set_transition_fast(
            "awareness.U", "awareness.A", ("contagion.S",), gamma
        )

        # set up weights of transitions for UA and infected
        cg.set_transition_fast(
            "awareness.U", "awareness.A", ("contagion.I",), delta
        )

        # set up weights of transitions for UA and removed
        cg.set_transition_fast(
            "awareness.U", "awareness.A", ("contagion.R",), gamma
        )

        return cg

    @property
    def _compartmental_graph(self) -> CompartmentalGraph:
        """Compartmental model that defines allowed transitions and states."""
        return self.__comp_graph

    @property
    def _seed_selector(self) -> nd.seeding.RandomSeedSelector:
        """A method of selecting seed agents."""
        return self.__seed_selector

    def __str__(self) -> str:
        descr = f"{nd.utils.BOLD_UNDERLINE}\n"
        descr += f"SIR-UA Model\n"
        descr += self._compartmental_graph.__str__()
        descr += str(self._seed_selector)
        return descr

    def determine_initial_states(
        self, net: nd.MultilayerNetwork
    ) -> list[NetworkUpdateBuffer]:
        if not net.is_multiplex():
            raise ValueError("This model works only with multiplex networks!")

        budget = self._compartmental_graph.get_seeding_budget_for_network(net)
        nodes_ranking = self._seed_selector.nodewise(net)
        initial_states = []

        # set initial states in contagion process/layer
        for node_position, node_id in enumerate(nodes_ranking["contagion"]):
            if node_position < budget["contagion"]["I"]:
                node_initial_state = "I"
            else:
                node_initial_state = "S"
            initial_states.append(
                nd.models.NetworkUpdateBuffer(
                    node_id, "contagion", node_initial_state
                )
            )

        # set initial states in awareness process/layer
        for node_position, node_id in enumerate(nodes_ranking["awareness"]):
            if node_position < budget["awareness"]["A"]:
                node_initial_state = "A"
            else:
                node_initial_state = "U"
            initial_states.append(
                nd.models.NetworkUpdateBuffer(
                    node_id, "awareness", node_initial_state
                )
            )

        return initial_states

    @staticmethod
    def flip_a_coin(prob_success: float) -> bool:
        result = np.random.choice([0, 1], p=[1 - prob_success, prob_success])
        if result == 1:
            return True
        return False

    def agent_evaluation_step(
        self, agent: int, layer_name: str, net: nd.MultilayerNetwork
    ) -> str:
        layer_graph: nx.Graph = net[layer_name]

        # get possible transitions for state of the node
        current_state = layer_graph.nodes[agent]["status"]
        transitions = self._compartmental_graph.get_possible_transitions(
            net.get_actor(agent).states_as_compartmental_graph(), layer_name
        )

        # if there is no possible transition don't do anything
        if len(transitions) == 0:
            return current_state

        # if transition doesn't rely on interacitons with neighbours (i.e. I->R)
        if layer_name == "contagion" and current_state == "I":
            new_state = "R"
            if self.flip_a_coin(transitions[new_state]):
                return new_state

        # otherwise iterate through neighours
        else:
            for neighbour in nx.neighbors(layer_graph, agent):
                new_state = layer_graph.nodes[neighbour]["status"]
                if new_state in transitions and self.flip_a_coin(
                    transitions[new_state]
                ):
                    return new_state

        return current_state

    def network_evaluation_step(
        self, net: nd.MultilayerNetwork
    ) -> list[NetworkUpdateBuffer]:
        new_states = []
        for layer_name, layer_graph in net.layers.items():
            for node in layer_graph.nodes():
                new_state = self.agent_evaluation_step(node, layer_name, net)
                layer_graph.nodes[node]["status"] = new_state
                new_states.append(
                    nd.models.NetworkUpdateBuffer(node, layer_name, new_state)
                )
        return new_states

    def get_allowed_states(
        self, net: nd.MultilayerNetwork
    ) -> dict[str, tuple[str, ...]]:
        return self._compartmental_graph.get_compartments()


# 3rd section

net = nd.MultilayerNetwork.from_nx_layer(
    nx.karate_club_graph(), ["contagion", "awareness"]
)
print(net)

# 4th section

model = SIR_UAModel(
    alpha=0.19,
    alpha_prime=0.0665,
    beta=0.1,
    gamma=0.01,
    delta=0.71,
    ill_seeds=5,
    aware_seeds=5,
)
print(model)

# 5th section

experiment = nd.Simulator(model, net)
run_logs = experiment.perform_propagation(n_epochs=10)

# 6th section

run_logs.report(path="results", visualisation=True)

# 7th section

run_logs.get_aggragated_logs()

# 8th section

run_logs.get_detailed_logs()
