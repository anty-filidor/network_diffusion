import os
import string
from typing import Any, Dict, List


# TODO - json
def read_mlx(file_path: str) -> Dict[str, List[Any]]:
    """
    Handler of MLX files for the MultilayerNetwork class.

    :param file_path: path to file

    :return: a dictionary with network to create class
    """
    # initialise empty containers
    net_dict = {}
    tab: List[Any] = []
    name = "foo"

    with open(file_path, "r") as file:
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
    Checks out if given directory exists and if doesn't it creates it.

    :param dest_path: absolute path to create folder
    """
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
