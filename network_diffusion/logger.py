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

"""Functions for logging experiment results."""

# pylint: disable=W0141
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_diffusion.utils import BOLD_UNDERLINE, THIN_UNDERLINE


class Logger:
    """Store and processes logs acquired during performing Simulator."""

    def __init__(
        self, model_description: str, network_description: str
    ) -> None:
        """
        Construct object.

        :param model_description: description of the model (i.e.
            BaseModel.__str__()) which is used for saving in logs
        :param network_description: description of the network (i.e.
            MultilayerNetwork.__str__()) which is used for saving in logs
        """
        self._model_description = model_description
        self._network_description = network_description

        # stores data of network global state in each epoch
        self._global_stats: List[Dict[str, Any]] = []
        self._global_stats_converted: Dict[str, Any] = {}

        # stores data of each of nodes that changed their state in each epoch
        self._local_stats: Dict[int, List[Dict[str, str]]] = {}

    def add_global_stat(self, log: Dict[str, Any]) -> None:
        """
        Add raw log from single epoch to the object.

        :param log: raw log (i.e. a single call of
            MultilayerNetwork.get_states_num())
        """
        self._global_stats.append(log)

    def add_local_stat(self, epoch: int, stats: List[Dict[str, str]]) -> None:
        """Add local log from single epoch to the object."""
        self._local_stats[epoch] = stats

    def convert_logs(
        self, model_parameters: Dict[str, Tuple[str, ...]]
    ) -> None:
        """
        Convert raw logs into pandas dataframe.

        Used after finishing aggregation of logs. It fulfills self._stats.

        :param model_parameters: parameters of the propagation model to store
        """
        # initialise container for splatted data
        self._global_stats_converted = {
            k: pd.DataFrame(columns=model_parameters[k])
            for k in model_parameters.keys()
        }

        # fill containers
        for epoch in self._global_stats:
            for layer, vals in epoch.items():
                self._global_stats_converted[layer] = pd.concat(
                    [
                        self._global_stats_converted[layer],
                        pd.DataFrame(dict(vals), index=[0]),
                    ],
                    ignore_index=True,
                )

        # change NaN values to 0 and all values to integers
        for layer, vals in self._global_stats_converted.items():
            self._global_stats_converted[layer] = vals.fillna(0).astype(int)

    def __str__(self) -> str:
        return str(self._global_stats_converted)

    def plot(self, to_file: bool = False, path: Optional[str] = None) -> None:
        """
        Plot out visualisation of performed experiment.

        :param to_file: flag, if true save figure to file, otherwise it
            is plotted on screen
        :param path: path to save figure
        """
        fig = plt.figure()

        for i, layer in enumerate(self._global_stats_converted, 1):
            ith_axis = fig.add_subplot(len(self._global_stats_converted), 1, i)
            self._global_stats_converted[layer].plot(ax=ith_axis, legend=True)
            ith_axis.set_title(layer)
            ith_axis.legend(loc="upper right")
            ith_axis.set_ylabel("Nodes")
            if i == 1:
                y_tics_num = self._global_stats_converted[layer].iloc[0].sum()
            ith_axis.set_yticks(np.arange(0, y_tics_num + 1, 20))
            if i == len(self._global_stats_converted):
                ith_axis.set_xlabel("Epoch")
            ith_axis.grid()

        plt.tight_layout()
        if to_file:
            plt.savefig(f"{path}/visualisation.png", dpi=200)
        else:
            plt.show()

    def report(
        self,
        visualisation: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """
        Create report of experiment.

        It consists of report of the network, report of the model, record of
        propagation progress and optionally visualisation of the progress.

        :param visualisation: (bool) a flag, if true visualisation is being
            plotted
        :param path: (str) path to folder where report will be saved if not
            provided logs are printed out on the screen
        """
        if path is not None:
            Path(path).mkdir(exist_ok=True, parents=True)

            # save progress in propagation of each layer to csv file
            for stat in self._global_stats_converted:
                self._global_stats_converted[stat].to_csv(
                    path + "/" + stat + "_propagation_report.csv",
                    index_label="epoch",
                )

            # save loacal stats of each epoch
            with open(f"{path}/local_stats.json", "w", encoding="utf-8") as f:
                json.dump(self._local_stats, f)

            # save description of model to txt file
            with open(
                file=path + "/model_report.txt", mode="w", encoding="utf=8"
            ) as file:
                file.write(self._model_description)

            # save description of network to txt file
            with open(
                file=path + "/network_report.txt", mode="w", encoding="utf=8"
            ) as file:
                file.write(self._network_description)

            # save figure
            if visualisation:
                self.plot(to_file=True, path=path)

        else:
            print(self._network_description)
            print(self._model_description)
            print(f"{BOLD_UNDERLINE}\npropagation report\n{THIN_UNDERLINE}")
            for stat in self._global_stats_converted:
                print(stat, "\n", self._global_stats_converted[stat], "\n")
            print(BOLD_UNDERLINE)
            if visualisation:
                self.plot()

    def get_aggragated_logs(self) -> List[Dict[str, Any]]:
        """Get aggregated logs from the experiment as a list of dicts."""
        return self._global_stats

    def get_detailed_logs(self) -> Dict[int, List[Dict[str, str]]]:
        """Get detailed logs from the experiment as a dict of list of dicts."""
        return self._local_stats
