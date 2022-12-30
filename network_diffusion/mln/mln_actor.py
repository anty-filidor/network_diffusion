"""Contains funcitons to handle actors of the multilayer network."""

from typing import Dict, Tuple


class MLNetworkActor:
    """Dataclass that contain data of actor in the network."""

    def __init__(self, actor_id: str, layers_states: Dict[str, str]) -> None:
        """
        Initialise the object.

        :param actor_id: if of the actor
        :param layers_states: a dictionary keyed by layer names where the actor
            exists and valued by its state in the given layer
        """
        self.actor_id = actor_id
        self._layers_states = layers_states
    
    def __str__(self) -> str:
        return f"actor id: {self.actor_id}, layers and states: {self._layers_states}"
    
    # def __cmp__(self, __o: "MLNetworkActor") -> int:
    #     if __o.actor_id == self.actor_id and __o._layers_states == __o._layers_states:
    #         return 0
    #     elif __o.actor_id > self.actor_id:
    #         return 1
    #     elif __o.actor_id < self.actor_id:
    #         return -1
    #     return False

    @property
    def layers(self) -> Tuple[str, ...]:
        """Get network layers where actor exists."""
        return tuple(self._layers_states.keys())

    @property
    def states(self) -> Tuple[str, ...]:
        """Get actor's states for  where actitor exists."""
        return tuple(self._layers_states.values())

    @states.setter
    def states(self, updated_states: Dict[str, str]) -> None:
        """Set actor's states for layers where it exists."""
        for layer_name, new_state in updated_states.items():
            assert layer_name in self._layers_states
            self._layers_states[layer_name] = new_state
