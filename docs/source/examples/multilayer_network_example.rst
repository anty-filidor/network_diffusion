
==============================
Class ``nd.MultilayerNetwork``
==============================

In many real-world systems, the same set of entities can interact in different
contexts or through different types of relationships. A multilayer network
provides a way to represent such heterogeneous systems by organising
interactions into separate layers while keeping the underlying actors shared.

Formally, a multilayer network can be defined as follows.

.. admonition:: Multilayer network

   A **multilayer network** is defined as a quadruple

   .. math::

      G = ([n], [\ell], V, E),

   where:

   * :math:`[n] = \{1, 2, \ldots, n\}` denotes a set of **actors**,
   * :math:`[\ell] = \{1, 2, \ldots, \ell\}` denotes a set of **layers**,
   * :math:`V \subseteq [n] \times [\ell]` is the set of **nodes**,
   * :math:`E \subseteq \bigcup_{i \in [\ell]} (V_i \times V_i)` is the set of **edges**.

   Each node is represented as a pair :math:`(a, i)`, indicating appearance of
   the actor :math:`a` in layer :math:`i`. Also in such a system, edges
   connect nodes belonging to the same layer.

From this perspective, a multilayer network can be also understood as a
collection of graphs:

.. math::

   G_i = (V_i, E_i)

where each graph :math:`G_i` is **spanned on a subset of actors present in the
system** within the relationship (layer) :math:`i \in [\ell]`.

Example
-------

Consider a system describing interactions across several social media
platforms. Suppose we observe a group of people who maintain accounts on
different services such as LinkedIn, Twitter, or Facebook. In this setting,
the set :math:`[n]` represents the people active on the Internet, while
:math:`[\ell]` represents the social platforms.

Each user account corresponds to a node in the multilayer network. For
instance, a LinkedIn profile belonging to a particular person would be
represented as the node :math:`(a, i)`, where :math:`a` denotes the person
and :math:`i` corresponds to the LinkedIn layer.

Edges capture interactions between users within the same platform, such as
connections, follows, or messages. Consequently, each platform forms a
separate graph describing interactions specific to that service.

From this perspective, the multilayer network can be decomposed into
:math:`\ell` interdependent graphs, each corresponding to a different
social platform.

Using multilayer networks in ``network_diffusion``
==================================================

``nd.MultilayerNetwork`` extends the functionality of the ``networkx.Graph``
class to store and manipulate networks consisting of multiple relations
grouped into distinct layers. This is a fundamental structure in
``network_diffusion``.

Available data
==============
`Here <http://multilayer.it.uu.se/datasets.html>`_ is an example repository
containing multilayer network datasets that can be read from
``network_diffusion``.

Example of usage
================
Create multilayer networks in several ways.

1. By defining separate graphs and layer names

.. code-block:: python

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

2. By defining separate graphs and using concrete names for the layers

.. code-block:: python

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

3. By reading directly the `mpx` file

.. code-block:: python

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
=====================
``network_diffusion`` provides functions to read properties of multilayer
networks and centrality metrics tailored for such models (e.g., VoteRank,
PageRank, community detection routines). See the reference guide for further
information.
