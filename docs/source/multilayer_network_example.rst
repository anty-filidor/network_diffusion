
Module ``multilayer_network``
==============================

What is a multilayer network?
______________________________
Multilayer network is a class to extend the functionality of ``networkx.Graph``
library to store and manipulate multilayer networks, which are a fundamental
structure in the library. The module also allows to read the network from
``mpx`` text files, which store such structures.

Available data
______________
`Here <http://multilayer.it.uu.se/datasets.html>`_ is the exemplary repository
with multilayer networks, which can be read by ``network_diffusion``.

Example of usage
________________
Let's crete some multilayer networks in several ways.

1. By defining separate graphs and layer names::

    import networkx as nx
    import network_diffusion as nd

    layers = [
        nx.les_miserables_graph(),
        nx.les_miserables_graph(),
        nx.les_miserables_graph(),
    ]
    ml_network = nd.MultilayerNetwork.from_nx_layers(layers)

    print(ml_network)

.. code-block:: console

    ============================================
    network parameters
    --------------------------------------------
    general parameters:
        number of layers: 3
        number of actors: 77
        number of nodes: 231
        number of edges: 762

    layer 'layer_0' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731

    layer 'layer_1' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731

    layer 'layer_2' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731
    ============================================

2. By defining separate graphs and using concrete names for the layers::

    import networkx as nx
    import network_diffusion as nd

    layers = [
        nx.les_miserables_graph(),
        nx.les_miserables_graph(),
        nx.les_miserables_graph(),
    ]
    ml_network = nd.MultilayerNetwork.from_nx_layers(layers, ["l1", "l2", "l3"])

    print(ml_network)

.. code-block:: console

    ============================================
    network parameters
    --------------------------------------------
    general parameters:
        number of layers: 3
        number of actors: 77
        number of nodes: 231
        number of edges: 762

    layer 'l1' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731

    layer 'l2' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731

    layer 'l3' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731
    ============================================

3. By reading directly the `mpx` file::

    import network_diffusion as nd

    mpx_path = "network_diffusion/tests/data/florentine.mpx"
    ml_network = nd.MultilayerNetwork.from_mpx(mpx_path)

    print(ml_network)

.. code-block:: console

    ============================================
    network parameters
    --------------------------------------------
    general parameters:
        number of layers: 2
        number of actors: 15
        number of nodes: 26
        number of edges: 35

    layer 'marriage' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 15
        number of edges - 20
        average degree - 2.6667
        clustering coefficient - 0.16

    layer 'business' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 11
        number of edges - 15
        average degree - 2.7273
        clustering coefficient - 0.4333
    ============================================

Other functionalities
_____________________
``network_diffusion`` provides functions to read properties of multilayer
networks and centrality metrics tailored for such models (e.g., VoteRank,
PageRank, community detection routines). See the reference guide for further
information.
