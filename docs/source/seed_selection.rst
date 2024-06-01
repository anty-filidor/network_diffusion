==============================================
Seed selection classes for propagation models
==============================================

See dedicated `propagation model guide <propagation_model_example.html>`_ for
these functions.

Base structures for concrete seed selectors
===========================================

.. autoclass:: network_diffusion.seeding.base_selector.BaseSeedSelector
    :members:
    :private-members:
    :show-inheritance:

Concrete seed selectors
========================

.. autoclass:: network_diffusion.seeding.betweenness_selector.BetweennessSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.cbim_selector.CBIMSeedselector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.cim.CIMSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.closeness_selector.ClosenessSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.degreecentrality_selector.DegreeCentralityDiscountSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.driveractor_selector.DriverActorSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.katz_selector.KatzSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.kppshell_selector.KPPShellSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.kshell_selector.KShellMLNSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.kshell_selector.KShellSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.mocking_selector.MockingActorSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.neighbourhoodsize_selector.NeighbourhoodSizeDiscountSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.pagerank_selector.PageRankMLNSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.pagerank_selector.PageRankSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.random_selector.RandomSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.voterank_selector.VoteRankMLNSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: network_diffusion.seeding.voterank_selector.VoteRankSeedSelector
    :members:
    :undoc-members:
    :show-inheritance:
