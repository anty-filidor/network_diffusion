===============
Network models
===============

The library supports multilayer networks, temporal networks, as well as their composition (i.e.,
time series of multilayer networks). Below is a list of classes and functions that enable the
modelling and manipulation of these complex network structures.

Operations on multilayer networks
=================================

See dedicated `multilayer guide <multilayer_network_example.html>`_ for these functions.
The library is equipped with two multilayer networks working out-of-the-box:
``toy_network_piotr`` and ``toy_network_cim`` (:ref:`aux-mln-label`).

Class ``MLNetworkActor``
++++++++++++++++++++++++

Implemented in ``network_diffusion.mln.actor``.

.. autoclass:: network_diffusion.mln.actor.MLNetworkActor
   :members:

Class ``MultilayerNetwork``
+++++++++++++++++++++++++++

Implemented in ``network_diffusion.mln.mlnetwork``.

.. autoclass:: network_diffusion.mln.mlnetwork.MultilayerNetwork
   :members:

.. _aux-mln-label:

Auxiliary functions for operations on ``MultilayerNetwork``
-----------------------------------------------------------

Implemented in ``network_diffusion.mln.functions``.

.. automodule:: network_diffusion.mln.cbim
   :members:

.. automodule:: network_diffusion.mln.centrality_discount
   :members:

.. automodule:: network_diffusion.mln.driver_actors
   :members:

.. automodule:: network_diffusion.mln.functions
   :members:

.. automodule:: network_diffusion.mln.kppshell
   :members:

Class ``MultilayerNetworkTorch``
++++++++++++++++++++++++++++++++

Implemented in ``network_diffusion.mln.mlnetwork_torch``.

.. autoclass:: network_diffusion.mln.mlnetwork_torch.MultilayerNetworkTorch
   :members:

Operations on temporal networks
===============================

A ``TemporalNetwork`` is an ordered sequence of ``MultilayerNetwork`` objects. In a base scenario,
one can obtain a classic temporal network by having a chain of one-layered ``MultilayerNetwork``.

Please bear in mind that to perform simulations on ``TemporalNetwork`` objects, they must have
the same set of actors on each snapshot!

The library is equipped with an exemplar temporal-multilayer network: ``l2_course_net``
(:ref:`aux-tpn-label`).

Class ``TemporalNetwork``
+++++++++++++++++++++++++

Implemented in ``network_diffusion.tpn``.

.. autoclass:: network_diffusion.tpn.tpnetwork.TemporalNetwork
   :members:

.. _aux-tpn-label:

Module ``l2_course_net``
++++++++++++++++++++++++

Implemented in ``network_diffusion.tpn.l2_course_net``.

.. automodule:: network_diffusion.tpn.l2_course_net
   :members:
