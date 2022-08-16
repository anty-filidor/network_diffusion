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

"""Functions for logging experiments."""

# pylint: disable=W0141

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_diffusion.utils import create_directory


class ExperimentLogger:
    """Store and processes logs acquired during performing MultiSpreading."""

    def __init__(
        self, model_description: str, network_description: str
    ) -> None:
        """
        Construct object.

        :param model_description: description of the model (i.e.
            PropagationModel.describe()) which is used for saving in logs
        :param network_description: description of the network (i.e.
            MultiplexNetwork.describe()) which is used for saving in logs
        """
        self._model_description = model_description
        self._network_description = network_description
        self._raw_stats: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {}

    def _add_log(self, log: Dict[str, Any]) -> None:
        """
         Add raw log from single epoch to the object.

        :param log: raw log (i.e. a single call of
            MultiplexNetwork.get_nodes_states())
        """
        self._raw_stats.append(log)

    def _convert_logs(
        self, model_parameters: Dict[str, Tuple[Dict[str, Any]]]
    ) -> None:
        """
        Convert raw logs into pandas dataframe.

        Used after finishing aggregation of logs. It fulfills self._stats.

        :param model_parameters: parameters of the propagation model to store
        """
        # initialise container for splatted data
        self._stats = {
            k: pd.DataFrame(columns=model_parameters[k])
            for k in model_parameters.keys()
        }

        # fill containers
        for epoch in self._raw_stats:
            for layer, vals in epoch.items():
                self._stats[layer] = pd.concat(
                    [self._stats[layer], pd.DataFrame(dict(vals), index=[0])],
                    ignore_index=True,
                )

        # change NaN values to 0 and all values to integers
        for layer, vals in self._stats.items():
            self._stats[layer] = vals.fillna(0).astype(int)

    def __str__(self) -> str:
        return str(self._stats)

    def plot(self, to_file: bool = False, path: Optional[str] = None) -> None:
        """
        Plot out visualisation of performed experiment.

        :param to_file: flag, if true save figure to file, otherwise it
            is plotted on screen
        :param path: path to save figure
        """
        fig = plt.figure()

        for i, layer in enumerate(self._stats, 1):
            ith_axis = fig.add_subplot(len(self._stats), 1, i)
            self._stats[layer].plot(ax=ith_axis, legend=True)
            ith_axis.set_title(layer)
            ith_axis.legend(loc="upper right")
            ith_axis.set_ylabel("Nodes")
            if i == 1:
                y_tics_num = self._stats[layer].iloc[0].sum()
            ith_axis.set_yticks(np.arange(0, y_tics_num + 1, 20))
            if i == len(self._stats):
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
        to_file: bool = False,
        path: Optional[str] = None,
    ) -> None:
        """
        Create report of experiment.

        It consists of report of the network, report of the model, record of
        propagation progress and optionally visualisation of the progress.

        :param visualisation: (bool) a flag, if true visualisation is being
            plotted
        :param to_file: a flag, if true report is saved in files,
            otherwise it is printed out on the screen
        :param path: (str) path to folder where report will be saved
        """
        if to_file:
            if path is None:
                raise AttributeError(
                    "If to_file is True, then path cannot be None!"
                )
            # create directory from given path
            create_directory(path)
            # save progress in propagation of each layer to csv file
            for stat in self._stats:
                self._stats[stat].to_csv(
                    path + "/" + stat + "_propagation_report.csv",
                    index_label="epoch",
                )
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
            print(
                "============================================\n"
                "propagation report\n"
                "--------------------------------------------"
            )
            for stat in self._stats:
                print(stat, "\n", self._stats[stat], "\n")
            print("============================================")
            if visualisation:
                self.plot()
