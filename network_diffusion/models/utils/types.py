"""Definition of the aux types used in the library."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class NetworkUpdateBuffer:
    """Auxiliary class to keep info about nodes that needs to be udated."""

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
