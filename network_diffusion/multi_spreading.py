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

"""Functions for the phenomena spreading definition."""
from tqdm import tqdm

from network_diffusion.experiment_logger import ExperimentLogger
from network_diffusion.models.base_model import BaseModel
from network_diffusion.multilayer_network import MultilayerNetwork


class MultiSpreading:
    """Perform experiment defined by PropagationModel on MultiLayerNetwork."""

    def __init__(self, model: BaseModel, network: MultilayerNetwork) -> None:
        """
        Construct an object.

        :param model: model of propagation which determines how experiment
            looks like
        :param network: a network which is being examined during experiment
        """
        assert (
            network.layers.keys()
            == model.compartments.get_compartments().keys()
        ), (
            "Layer names in network should be the same as layer names in "
            "propagation model"
        )
        self._model = model
        self._network = network

    def perform_propagation(self, n_epochs: int) -> ExperimentLogger:
        """
        Perform experiment on given network and given model.

        It saves logs in ExperimentLogger object which can be used for further
        analysis.

        :param n_epochs: number of epochs to do experiment
        :return: logs of experiment stored in special object
        """
        logger = ExperimentLogger(
            str(self._model),
            self._network._get_description_str(),
        )

        # set and add logs from initialising states
        self._model.set_initial_states(self._network)
        logger._add_log(self._network.get_nodes_states())

        # iterate through epochs
        progress_bar = tqdm(range(n_epochs))
        for epoch in progress_bar:
            progress_bar.set_description_str(f"Processing epoch {epoch}")

            # do a forward step
            nodes_to_update = self._model.network_evaluation_step(
                self._network
            )
            epoch_json = self._model.update_network(self._network, nodes_to_update)

            # add logs from current epoch
            logger._add_log(self._network.get_nodes_states())
            logger._local_stats[epoch] = epoch_json

        # convert logs to dataframe
        logger._convert_logs(self._model.compartments.get_compartments())

        return logger
