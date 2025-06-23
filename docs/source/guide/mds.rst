======================
Minimal Dominating Set
======================

`network_diffusion` provides functions for obtaining a Minimal Dominating Set (MDS) of actors
in a multilayer network. This concept, originating from the domain of controllability, introduces
the notion of a set through which all nodes (i.e. actor representatives on individual layers) can be
reached via direct impulses from the MDS.

For reference, see:
https://iopscience.iop.org/article/10.1088/1367-2630/14/7/073005/pdf
https://arxiv.org/abs/2502.15236

Methods for obtaining the MDS
++++++++++++++++++++++++++++++

.. automodule:: network_diffusion.mln.mds.greedy_search
   :members:

.. automodule:: network_diffusion.mln.mds.local_improvement
   :members:
   :exclude-members: LocalImprovement
