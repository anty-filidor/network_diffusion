# Copyright (c) 2022 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""Network Diffusion is a package for simulating spreading phenomena."""

# flake8: noqa

import importlib.metadata

from network_diffusion import mln, models, seeding, tpn
from network_diffusion.logger import Logger
from network_diffusion.mln.actor import MLNetworkActor
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.simulator import Simulator
from network_diffusion.tpn.tpnetwork import TemporalNetwork

__version__ = importlib.metadata.version("network_diffusion")
