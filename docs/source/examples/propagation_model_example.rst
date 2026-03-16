==================
Propagation models
==================

What is a propagation model?
============================
In ``network_diffusion``, a propagation model is considered as one or several
phenomena acting in the network. To create a spreading model, we must extend
the abstract base class ``nd.models.BaseModel`` by implementing the following
methods, which define the mechanism of diffusion:

* ``determine_initial_states``,
* ``agent_evaluation_step``,
* ``network_evaluation_step``.

Additionally, the following class attributes must be provided:

* ``_compartmental_graph``,
* ``_seed_selector``,

and two auxiliary methods for reporting: ``__str__`` and
``get_allowed_states``.

It is important to note that the spreading model does not treat a network as
an internal parameter. Networks are bound to models during experiment execution
in the ``nd.Simulator`` class.

``nd.models.BaseModel.determine_initial_states``
________________________________________________
This method determines the initial state of the network before the start of the
simulation. At this stage, the diffusion seeds are also chosen. The method
returns
a list of ``NetworkUpdateBuffer`` structures, which contain the necessary
information for updating the network state (specifically, the agent's ID and
states in each layer).

``nd.models.BaseModel.agent_evaluation_step``
_____________________________________________
This method is called during each simulation step for each agent individually.
It takes the agent's ID, the ID of the layer for which the evaluation takes place,
and the network itself. It determines the state the agent will adopt in the next step of the
experiment. This is done by reading the set of possible states into which the
agent can fall, and by iteratively attempting (for example) to adopt the
neighbour's state with a probability derived from the ``_compartmental_graph``.
The output
of this function is the new state of an agent in the given network layer.

``nd.models.BaseModel.network_evaluation_step``
_______________________________________________
This method is responsible for evaluating the entire network in a single
experiment step. It involves iterating over each agent to determine its new state
according to the method ``agent_evaluation_step``. The output is a list of
``NetworkUpdateBuffer`` structures.

``nd.models.BaseModel._seed_selector``
______________________________________
This property ranks agents of the network. Then, the ranking is used to set
seeds of diffusion, i.e., a given fraction of agents is chosen from the beginning
of the list. For details, see `a reference <seed_selection.html>` of available
seed selectors.

``nd.models.BaseModel._compartmental_graph``
____________________________________________
This property is used to store all possible states of the agents and allowed
transitions between them. For details, see
`a guide <compartmental_graph_example.html>`_ of using this class.

Auxiliary abstract methods
__________________________
To provide comprehensive experiment logging, a user needs to implement two
functions. The first one - ``__str__`` should print a general description of the
model with compartments and allowed transitions (which can be readily derived from a
property ``_compartmental_graph`` ). The second method, ``get_allowed_states``
is used in the process of creating a simulation report. It is supposed to return
a dictionary with a list of network layers where each of phenomena took place
and allowed states in each layer.

Concrete spreading models
=========================
``network_diffusion`` also provides out-of-the-box spreading models (like LTM or
ICM) adapted to multilayer networks. For details, see
`a reference <spreading_models.html>`_.
