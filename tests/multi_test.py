from os import getcwd

import networkx as nx

from network_diffusion.multi_spreading import MultiSpreading
from network_diffusion.multilayer_network import MultilayerNetwork
from network_diffusion.propagation_model import PropagationModel
from network_diffusion.multi_spreading import MultiSpreading
from os import getcwd

# first example
"""
# initialise multilayer network from nx prefedined network
network = MultilayerNetwork()
names = ["illness", "awareness", "vaccination"]
network.load_layer_nx(nx.les_miserables_graph(), names)

# initialise propagation model and set possible transitions with probabilities
model = PropagationModel()
phenomenas = [("S", "I", "R"), ("UA", "A"), ("UV", "V")]
for l, p in zip(names, phenomenas):
    model.add(l, p)
model.compile(background_weight=0.005)

model.set_transition_fast(
    "illness.S", "illness.I", ["vaccination.UV", "awareness.UA"], 0.9
)
model.set_transition_fast(
    "illness.S", "illness.I", ["vaccination.V", "awareness.A"], 0.05
)
model.set_transition_fast(
    "illness.S", "illness.I", ["vaccination.UV", "awareness.A"], 0.2
)
model.set_transition_fast(
    "illness.I", "illness.R", ["vaccination.UV", "awareness.UA"], 0.1
)
model.set_transition_fast(
    "illness.I", "illness.R", ["vaccination.V", "awareness.A"], 0.7
)
model.set_transition_fast(
    "illness.I", "illness.R", ["vaccination.UV", "awareness.A"], 0.3
)

model.set_transition_fast(
    "vaccination.UV", "vaccination.V", ["awareness.A", "illness.S"], 0.03
)
model.set_transition_fast(
    "vaccination.UV", "vaccination.V", ["awareness.A", "illness.I"], 0.01
)

model.set_transition_fast(
    "awareness.UA", "awareness.A", ["vaccination.UV", "illness.S"], 0.05
)
model.set_transition_fast(
    "awareness.UA", "awareness.A", ["vaccination.V", "illness.S"], 1
)
model.set_transition_fast(
    "awareness.UA", "awareness.A", ["vaccination.UV", "illness.I"], 0.2
)

model.set_transition_canonical(
    "illness",
    (
        ("awareness.UA", "illness.R", "vaccination.UV"),
        ("awareness.UA", "illness.I", "vaccination.UV"),
    ),
    0.7,
)

# initialise starting parameters of propagation in network
phenomenas = {
    "illness": (65, 10, 2),
    "awareness": (60, 17),
    "vaccination": (70, 7),
}

# perform propagation experiment
experiment = MultiSpreading(model, network)
experiment.set_initial_states(phenomenas)
logs = experiment.perform_propagation(200)

# save experiment results
logs.report(to_file=False, path=None, visualisation=True)
"""

# second example
"""
# initialise multilayer network from mlx file
network = MultilayerNetwork()
network.load_mlx('/Users/michal/PycharmProjects/network_diffusion/tests/networks/florentine.mpx')

# initialise propagation model and set possible transitions with probabilities
model = PropagationModel()
phenomenas = [('S', 'I'), ('UV', 'V')]
for l, p in zip(network.get_nodes_states(), phenomenas):
    model.add(l, p)
model.compile(background_weight=0.01)

model.set_transition_fast('marriage.S', 'marriage.I', ['business.UV'], 0.9)
model.set_transition_fast('marriage.S', 'marriage.I', ['business.V'], 0.3)

model.set_transition_fast('business.UV', 'business.V', ['marriage.S'], 0.05)
model.set_transition_fast('business.UV', 'business.V', ['marriage.I'], 0.03)

# initialise starting parameters of propagation in network
phenomenas = {'marriage': (12, 3), 'business': (10, 1)}

# perform propagation experiment
experiment = MultiSpreading(model, network)
experiment.set_initial_states(phenomenas)
logs = experiment.perform_propagation(200)
logs.report(to_file=True, path=getcwd()+'/results', visualisation=True)
"""

# third example
"""
# initialise multilayer network from mlx file
network = MultilayerNetwork()
network.load_mlx('/Users/michal/PycharmProjects/network_diffusion/tests/networks/aucs.mpx')
network.describe()

# initialise propagation model and set possible transitions with probabilities
model = PropagationModel()
phenomenas = [('S', 'I', 'R'), ('UA', 'A'), ('UV', 'V'), ('S', 'I', 'R'), ('UV', 'V')]
for l, p in zip(network.get_nodes_states(), phenomenas):
    model.add(l, p)
model.compile(background_weight=0.2)
model.set_transitions_in_random_edges([[0.4, 0.5], [0.3, 0.2, 0.1], [0.9], [0.8, 0.6], [0.7]])
model.describe()

# initialise starting parameters of propagation in network
phenomenas = {'facebook': (10, 3, 19), 'lunch': (59, 1), 'coauthor': (23, 2), 'leisure': (40, 5, 2), 'work': (1, 59)}

# perform propagation experiment
experiment = MultiSpreading(model, network)
experiment.set_initial_states(phenomenas)
logs = experiment.perform_propagation(200)
logs.report(to_file=True, path=getcwd()+'/results', visualisation=True)
"""
