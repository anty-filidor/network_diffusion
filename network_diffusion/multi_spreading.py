from random import shuffle


class MultiSpreading:
    def __init__(self, model, network):
        assert network.layers.keys() == model.get_model_hyperparams().keys(), \
            'Layer names in network should be the same as layer names in propagation model'
        self.model = model
        self.network = network

    def set_initial_states(self, states_seeds, track_changes=False):
        """
        This method prepares network to be used during simulation. It changes status of certain nodes following given
        seeds in function argument
        :param states_seeds: dictionary with numbers of nodes to be initialise in each layer of network, i.e.
        {'illness': (75, 2, 0), 'awareness': (60, 17), 'vaccination': (70, 7)} for this: [illness : ('S', 'I', 'R')
        awareness : ('UA', 'A')vvaccination : ('UV', 'V')] model of propagation
        Note, that values in dictionary says how many nodes should have status <x> at the beginning of the experiment
        :param track_changes: flag, if true changes are being printed out
        """

        model_hyperparams = self.model.get_model_hyperparams()

        # check if argument has good shape
        ass_states_arg = [len(s) for s in states_seeds.values()]
        ass_states_net = [len(s) for s in model_hyperparams.values()]
        assert ass_states_net == ass_states_arg, \
            'Shape of argument {} should be the same as shape of states in propagation model {}'\
            .format(ass_states_arg, ass_states_net)

        # check if given argument has good keys
        assert states_seeds.keys() == model_hyperparams.keys(), \
            'Layer names in argument should be the same as layer names in propagation model'

        # check if given argument has good values
        ass_states_sum = [sum(s[1]) == len(self.network.layers[s[0]].nodes()) for s in states_seeds.items()]
        for ssum in ass_states_sum:
            assert ssum, 'Sum of size of states for each layer should be equal to number of its nodes!'

        # set initial states in each layer of network
        for name, layer in self.network.layers.items():
            states = model_hyperparams[name]
            vals = states_seeds[name]
            N = len(layer.nodes())
            if track_changes: print(name, states, vals, N)

            # shuffle nodes
            nodes = [*layer.nodes()]
            shuffle(nodes)

            # set ranges
            ranges = [sum(vals[:x]) for x in range(len(vals))]
            ranges.append(N)
            ranges = [(i, j) for i, j in zip(ranges[:-1], ranges[1:])]
            if track_changes: print(ranges)

            # append states to nodes
            for i in range(len(ranges)):
                pair = ranges[i]
                state = states[i]
                if track_changes: print(pair, state)
                for index in range(pair[0], pair[1]):
                    if track_changes: print(index, nodes[index], layer.node[nodes[index]])
                    layer.node[nodes[index]]['status'] = state
                    if track_changes: print(index, nodes[index], layer.node[nodes[index]])

    def perform_propagation(self, n_epochs):
        for _ in range(n_epochs):
            for layer_name, layer_graph in self.network.layers.items():
                print(layer_name)
                print(layer_graph.node['Napoleon'])
                print(network.get_neighbours_attributes('Napoleon', layer_name))




import networkx as nx
from multiplex_network import MultiplexNetwork
from propagation_model import PropagationModel

# initialise multiplex network
layers = [nx.les_miserables_graph(), nx.les_miserables_graph(), nx.les_miserables_graph()]
names = ['illness', 'awareness', 'vaccination']
network = MultiplexNetwork()
network.load_layers_nx(layers, names)
network.describe()

# initialise propagation model
model = PropagationModel()
phenomenas = [('S', 'I', 'R'), ('UA', 'A'), ('UV', 'V')]
for l, p in zip(names, phenomenas):
    model.add(l, p)
model.compile()
model.set_transitions_in_random_edges([[0.5, 0.5, 0.4, 0.9, 0.1], [1], [0.004, 0.54, 0.6, 0.9]])
model.describe()

# initialise start parameters of propagation
phenomenas = {'illness': (75, 2, 0), 'awareness': (60, 17), 'vaccination': (70, 7)}


# perform propagation
diffusion = MultiSpreading(model, network)
diffusion.set_initial_states(phenomenas)
network.describe()
# diffusion.perform_propagation(10)