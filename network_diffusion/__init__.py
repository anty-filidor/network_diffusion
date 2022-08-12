#!/usr/bin/env python3

# Copyright 2022 by Michał Czuba, Piotr Bródka. All Rights Reserved.
#
# This file is part of Network Diffusion.
#
# Network Diffusion is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Network Diffusion is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the  GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# Network Diffusion. If not, see <http://www.gnu.org/licenses/>.
# =============================================================================

"""Network Diffusion is package for simulating spreading phenomenas."""

from network_diffusion.multi_spreading import MultiSpreading  # noqa: F401
from network_diffusion.multilayer_network import (  # noqa: F401
    MultilayerNetwork,
)
from network_diffusion.propagation_model import PropagationModel  # noqa: F401

__version__ = "0.6.2.dev"
