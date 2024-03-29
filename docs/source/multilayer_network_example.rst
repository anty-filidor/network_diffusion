
Module  ``multilayer_network``
==============================


What is a multilayer network?
______________________________
Multilayer nNtwork is a class to extend functionality of ``networkx.Graph``
library to store and manipulate multilayer networks, which are a fundamental
structure in the library. Module also allows to read network from *mpx* text
files which stores such a structures.

Available data
______________
Here is an exemplar repository with multilayer networks:
`hub <http://multilayer.it.uu.se/datasets.html>`_, but you find them in many
other sited around Internet.

Example of usage
________________
Let's crete some multilayer networks in several ways.

1. By defining separate graphs and layer names::

    from network_diffusion.mln.mlnetwork import MultilayerNetwork
    import networkx as nx

    M = [nx.les_miserables_graph(), nx.les_miserables_graph(), nx.les_miserables_graph()]

    mpx = MultilayerNetwork.from_nx_layers(M)
    mpx.describe()

.. code-block:: console

    ============================================
    network parameters
    --------------------------------------------
    general parameters:
        number of layers: 3
        multiplexing coefficient: 1.0
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

2. By defining separate graphs and using default names of layers::

    from network_diffusion.mln.mlnetwork import MultilayerNetwork
    import networkx as nx
    M = [nx.les_miserables_graph(), nx.les_miserables_graph(), nx.les_miserables_graph()]

    mpx = MultilayerNetwork.from_nx_layer(M, ['A', 'B', 'C'])
    mpx.describe()

.. code-block:: console

    ============================================
    network parameters
    --------------------------------------------
    general parameters:
        number of layers: 3
        multiplexing coefficient: 1.0
    layer 'A' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731
    layer 'B' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731
    layer 'C' parameters:
        graph type - <class 'networkx.classes.graph.Graph'>
        number of nodes - 77
        number of edges - 254
        average degree - 6.5974
        clustering coefficient - 0.5731
    ============================================


3. By reading out mpx file::

    mpx = MultilayerNetwork.from_mpx('/my_project/monastery.mpx')
    mpx.describe()


.. code-block:: console

    ============================================
    network parameters
    --------------------------------------------
    general parameters:
        number of layers: 10
        multiplexing coefficient: 0.7778
    layer 'like1' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 55
        average degree - 6.1111
        clustering coefficient - 0.1732
    layer 'like2' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 57
        average degree - 6.3333
        clustering coefficient - 0.2923
    layer 'like3' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 56
        average degree - 6.2222
        clustering coefficient - 0.3603
    layer 'dislike' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 17
        number of edges - 47
        average degree - 5.5294
        clustering coefficient - 0.1213
    layer 'esteem' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 54
        average degree - 6.0
        clustering coefficient - 0.3222
    layer 'desesteem' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 17
        number of edges - 58
        average degree - 6.8235
        clustering coefficient - 0.2029
    layer 'positive_influence' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 53
        average degree - 5.8889
        clustering coefficient - 0.3537
    layer 'negative_influence' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 50
        average degree - 5.5556
        clustering coefficient - 0.1084
    layer 'praise' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 18
        number of edges - 39
        average degree - 4.3333
        clustering coefficient - 0.3048
    layer 'blame' parameters:
        graph type - <class 'networkx.classes.digraph.DiGraph'>
        number of nodes - 15
        number of edges - 41
        average degree - 5.4667
        clustering coefficient - 0.1133
    ============================================
