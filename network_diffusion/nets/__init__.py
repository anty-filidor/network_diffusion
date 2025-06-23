# Copyright (c) 2025 by Michał Czuba, Piotr Bródka.
#
# This file is a part of Network Diffusion.
#
# Network Diffusion is licensed under the MIT License. You may obtain a copy
# of the License at https://opensource.org/licenses/MIT
# =============================================================================

"""This module contains networks which can be used with the library."""

# flake8: noqa

from network_diffusion.nets import mln_generator
from network_diffusion.nets.l2_course_net import get_l2_course_net
from network_diffusion.nets.toy_network_cim import get_toy_network_cim
from network_diffusion.nets.toy_network_piotr import get_toy_network_piotr

# TODO: add unit tests
