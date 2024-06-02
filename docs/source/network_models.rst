===============
Network models
===============

Operations on multilayer networks
=================================

See dedicated `multilayer guide <multilayer_network_example.html>`_ for these
functions.

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


Operations on temporal networks
=================================

In the library TemporalNetwork is an ordered sequence of MultilayerNetworks.
In base scenario one can obtain classic temporal network by having a chain
of one-layered MultilayerNetworks.

Class ``TemporalNetwork``
+++++++++++++++++++++++++

Implemented in ``network_diffusion.tpn.tpnetwork``.

.. autoclass:: network_diffusion.tpn.tpnetwork.TemporalNetwork
   :members:
