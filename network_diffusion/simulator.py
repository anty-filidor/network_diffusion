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

"""Functions for composing and executing an experiment."""

import warnings
from typing import Callable, List, Optional, Tuple, Union

from tqdm import tqdm

from network_diffusion.logger import Logger
from network_diffusion.mln.mlnetwork import MultilayerNetwork
from network_diffusion.models.base_model import BaseModel
from network_diffusion.models.utils.types import NetworkUpdateBuffer
from network_diffusion.tpn.tpnetwork import TemporalNetwork


class Simulator:
    """Perform experiment defined by BaseModel on MultiLayerNetwork."""

    def __init__(
        self,
        model: BaseModel,
        network: Union[MultilayerNetwork, TemporalNetwork],
    ) -> None:
        """
        Construct an object.

        :param model: model of propagation which determines how experiment
            looks like
        :param network: a network which is being examined during experiment
        """
        self._model = model
        self._network = network
        self.stopping_counter = 0

    def _update_counter(
        self,
        new_states: List[NetworkUpdateBuffer],
        old_states: List[NetworkUpdateBuffer],
    ) -> None:
        """Update a counter of dead epochs."""
        if set(new_states) == set(old_states):
            self.stopping_counter += 1
        else:
            self.stopping_counter = 0

    def _create_iterator(
        self, n_epochs: int
    ) -> Tuple[Callable[[int], MultilayerNetwork], int]:
        """Create iterator through snapshots of the network."""
        if isinstance(self._network, MultilayerNetwork):
            return lambda x: self._network, n_epochs  # type: ignore
        elif isinstance(self._network, TemporalNetwork):
            optim_epochs_nb = len(self._network) - 1
            if n_epochs > optim_epochs_nb:
                warnings.warn(
                    f"Number of simulation epochs is higher than number of "
                    f"snaps - 1! Simulation will last for "
                    f"{optim_epochs_nb} epochs",
                    stacklevel=1,
                )
            elif n_epochs < optim_epochs_nb:
                warnings.warn(
                    "Number of simulation epochs is lesser than number of "
                    "snaps - 1! Simulation will not cover entire net",
                    stacklevel=1,
                )
                optim_epochs_nb = n_epochs
            return lambda x: self._network[x], optim_epochs_nb  # type: ignore
        raise AttributeError("Incorrect type of network!")

    @staticmethod
    def _verify_network(net: TemporalNetwork, n_epochs: int) -> None:
        """Verify if in each snapshot there is the same actor set."""
        actors_0 = {a.actor_id for a in net[0].get_actors()}
        for epoch in range(1, n_epochs + 1):
            actors_epoch = {a.actor_id for a in net[epoch].get_actors()}
            if len(actors_epoch.symmetric_difference(actors_0)):
                raise ValueError("Net should have one actors set in snaps!")

    def perform_propagation(
        self,
        n_epochs: int,
        patience: Optional[int] = None,
    ) -> Logger:
        """
        Perform experiment on given network and given model.

        It saves logs in Logger object which can be used for further
        analysis.

        :param n_epochs: number of epochs to do experiment
        :param patience: if provided experiment will be stopped when in
            "patience" (e.g. 4) consecutive epoch there was no propagation
        :return: logs of experiment stored in special object
        """
        if patience is not None and patience <= 0:
            raise ValueError("Patience must be None or integer > 0!")
        snap_iterator, n_epochs = self._create_iterator(n_epochs)
        logger = Logger(str(self._model), str(self._network))

        # determine initial states, in epoch 0
        initial_states = self._model.determine_initial_states(snap_iterator(0))
        initial_json = self._model.update_network(
            snap_iterator(0), initial_states
        )

        # log inintial state of the network
        logger.add_global_stat(self._model.get_states_num(snap_iterator(0)))
        logger.add_local_stat(0, initial_json)

        # if network is temporal verify its consistence
        if isinstance(self._network, TemporalNetwork):
            self._verify_network(self._network, n_epochs)

        # main simulation loop
        p_bar = tqdm(range(n_epochs), "experiment", leave=False, colour="blue")
        old_states = initial_states
        for epoch in p_bar:
            p_bar.set_description_str(f"Processing epoch {epoch}")

            # obtain structure of the network in current and next epoch
            curr_snap = snap_iterator(epoch)
            next_snap = snap_iterator(epoch + 1)

            # do a forward step and update network
            new_states = self._model.network_evaluation_step(curr_snap)
            epoch_json = self._model.update_network(next_snap, new_states)

            # add logs from current epoch
            logger.add_global_stat(self._model.get_states_num(next_snap))
            logger.add_local_stat(epoch + 1, epoch_json)

            # check if there is no progress and therefore stop simulation
            if patience:
                self._update_counter(new_states, old_states)
                if self.stopping_counter >= patience:
                    p_bar.set_description_str(
                        f"Experiment stopped - no progress in last "
                        f"{patience} epochs!"
                    )
                    break
                old_states = new_states

        # convert logs to dataframe
        logger.convert_logs(self._model.get_allowed_states(snap_iterator(0)))

        return logger
