===============
Reference guide
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
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Implemented in ``network_diffusion.mln.functions``.

.. automodule:: network_diffusion.mln.functions
   :members:



Propagation models
==================

See dedicated `propagation model guide <propagation_model_example.html>`_ for
these functions.

Base structures for concrete models
+++++++++++++++++++++++++++++++++++

.. autoclass:: network_diffusion.models.base_model.BaseModel
    :members:

.. autoclass:: network_diffusion.models.utils.compartmental.CompartmentalGraph
    :members:

Concrete propagation models
+++++++++++++++++++++++++++

Import from ``network_diffusion.models``.

.. autoclass:: network_diffusion.models.dsaa_model.DSAAModel
    :members:
    :show-inheritance:

.. autoclass:: network_diffusion.models.mlt_model.MLTModel
    :members:
    :show-inheritance:



Performing experiments
======================

See dedicated `multi spreading guide <simulator_example.html>`_ for these
functions.

.. automodule:: network_diffusion.logger
   :members:
   :undoc-members:

.. automodule:: network_diffusion.simulator
   :members:
   :undoc-members:



Seed selection classes for propagation models
==============================================

See dedicated `propagation model guide <propagation_model_example.html>`_ for
these functions.

Base structures for concrete seed selectors
+++++++++++++++++++++++++++++++++++++++++++

.. autoclass:: network_diffusion.seeding.base_selector.BaseSeedSelector
    :members:
    :private-members:
    :show-inheritance:

Concrete seed selectors
+++++++++++++++++++++++

.. automodule:: network_diffusion.seeding.degreecentrality_selector
    :members:
    :show-inheritance:

.. automodule:: network_diffusion.seeding.kshell_selector
    :members:
    :show-inheritance:

.. automodule:: network_diffusion.seeding.mocky_selector
    :members:
    :show-inheritance:

.. automodule:: network_diffusion.seeding.neighbourhoodsize_selector
    :members:
    :show-inheritance:

.. automodule:: network_diffusion.seeding.pagerank_selector
    :members:
    :show-inheritance:

.. automodule:: network_diffusion.seeding.random_selector
    :members:
    :show-inheritance:

.. automodule:: network_diffusion.seeding.voterank_selector
    :members:
    :show-inheritance:



Auxiliary methods
=================

.. automodule:: network_diffusion.utils
   :members:
