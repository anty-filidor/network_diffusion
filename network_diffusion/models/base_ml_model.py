# Copyright 2023 by Michał Czuba, Piotr Bródka. All Rights Reserved.
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

"""Definition of the base propagation model used in the library."""

from abc import ABC
from typing import Dict, List

from network_diffusion.models.base_model import BaseModel
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.utils.types import NetworkUpdateBuffer


class BaseMLModel(BaseModel, ABC):
    """Base abstract multilayer network propagation model."""

    @staticmethod
    def update_network(
        net: MultilayerNetwork,
        activated_nodes: List[NetworkUpdateBuffer],
    ) -> List[Dict[str, str]]:
        """
        Update the network global state by list of already activated nodes.

        :param net: network to update
        :param activated_nodes: already activated nodes
        """
        out_json = []
        for active_node in activated_nodes:
            net.layers[active_node.layer_name].nodes[active_node.node_name][
                "status"
            ] = active_node.new_state
            out_json.append(active_node.to_json())
        return out_json
