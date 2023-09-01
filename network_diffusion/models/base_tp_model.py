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
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.tpn.tpnetwork import TemporalNetwork


class BaseTPModel(BaseModel, ABC):
    """Base abstract temporal network propagation model."""

    @staticmethod
    def update_network(
        net: TemporalNetwork,
        agents: List[NetworkUpdateBuffer],
        snapshot_id: int
    ) -> List[Dict[str, str]]:
        """
        Update the network snapshot state by list of already activated nodes.

        :param net: network to update
        :param agents: agents with changed attributes
        :param snapshot_id: id of the snapshot to be updated
        """
        out_json = []
        for agent in agents:
            net.snaps[snapshot_id].nodes[agent.node_name]["status"] = agent.new_state
            out_json.append(agent.to_json())
        return out_json
