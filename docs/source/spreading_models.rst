================
Spreading models
================

See dedicated `propagation model guide <propagation_model_example.html>`_ for
these functions.

Base structures for concrete models
===================================

.. autoclass:: network_diffusion.models.base_model.BaseModel
    :members:

.. autoclass:: network_diffusion.models.utils.compartmental.CompartmentalGraph
    :members:

Concrete propagation models
===========================

Import from ``network_diffusion.models``.

.. autoclass:: network_diffusion.models.dsaa_model.DSAAModel
    :members:
    :show-inheritance:

.. autoclass:: network_diffusion.models.mlt_model.MLTModel
    :members:
    :show-inheritance:

.. autoclass:: network_diffusion.models.mic_model.MICModel
    :members:
    :show-inheritance:

.. autoclass:: network_diffusion.models.tne_model.TemporalNetworkEpistemologyModel
    :members:
    :show-inheritance:
