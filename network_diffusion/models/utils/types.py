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

"""Definition of the aux types used in the library."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True, eq=True)
class NetworkUpdateBuffer:
    """Auxiliary class to keep info about nodes that needs to be updated."""

    node_name: str
    layer_name: str
    new_state: str

    def __str__(self) -> str:
        return f"{self.layer_name}:{self.node_name}:{self.new_state}"

    def to_json(self) -> Dict[str, str]:
        """Return dict writable to JSON."""
        return {
            "layer_name": self.layer_name,
            "node_name": self.node_name,
            "new_state": self.new_state,
        }
