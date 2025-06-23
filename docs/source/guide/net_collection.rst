======================
Collection of networks
======================

The library provides a basic set of networks which can be read directly. Also
it contains a generator of multilayer networks according to Erdos=Renyi and
Preferential attachement models.

Networks shipped with the library
+++++++++++++++++++++++++++++++++

.. automodule:: network_diffusion.nets.toy_network_cim
   :members:

.. automodule:: network_diffusion.nets.toy_network_piotr
   :members:

.. _aux-tpn-label:

.. automodule:: network_diffusion.nets.l2_course_net
   :members:

Generators of artificial networks
+++++++++++++++++++++++++++++++++

.. autoclass:: network_diffusion.nets.mln_generator.MultilayerERGenerator
   :members:

.. autoclass:: network_diffusion.nets.mln_generator.MultilayerPAGenerator
   :members:

.. autofunction:: network_diffusion.nets.mln_generator.generate
