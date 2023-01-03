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

"""Functions for the auxiliary operations."""


import os
import pathlib
import string
from typing import Any, Dict, List

BOLD_UNDERLINE = "============================================"
THIN_UNDERLINE = "--------------------------------------------"


def read_mpx(file_path: str) -> Dict[str, List[Any]]:
    """
    Handle MPX file for the MultilayerNetwork class.

    :param file_path: path to file

    :return: a dictionary with network to create class
    """
    # initialise empty containers
    net_dict = {}
    tab: List[Any] = []
    name = "foo"

    with open(file=file_path, mode="r", encoding="utf-8") as file:
        line = file.readline()

        # omit trash
        while line and line[0] != "#":
            line = file.readline()

        # read pure data
        while line:
            # if line contains title of new division
            if line[0] == "#":
                # if this is a special line - type
                if "#TYPE".lower() in line.lower():
                    net_dict.update(
                        {"type": [line[6:-1]]}
                    )  # '6:-1' to save the name of type
                # else if it is a normal division
                else:
                    net_dict.update({name: tab})
                    name = line[1:-1].lower()  # omitting '#' and '\n'
                    tab = []
            line = file.readline()
            # don't save line with only whitespaces or if line contains a
            # title of new division
            if not line.isspace() and "#" not in line:
                line = line.translate(
                    {ord(char): None for char in string.whitespace}
                )
                tab.append(line.split(","))

        # append last line to dictionary
        net_dict.update({name: tab[:-1]})
    del net_dict["foo"]

    return net_dict


def create_directory(dest_path: str) -> None:
    """
    Check out if given directory exists and if doesn't it creates it.

    :param dest_path: absolute path to create folder
    """
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)


def get_absolute_path() -> str:
    """Get absolute path of library."""
    return str(pathlib.Path(__file__).parent)
