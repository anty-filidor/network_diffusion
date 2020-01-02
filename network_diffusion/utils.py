import string
import os.path
from os import path
from functools import wraps

def read_mlx(file_path):
    """
    This function reads multilayer network from mlx file to the dictionary of arrays which is then readed by multilayer
    network class

    :param file_path: path to file
    :return: a dictionary with network to create class
    """

    # initialise empty containers
    net_dict = {}
    tab = []
    name = 'foo'

    with open(file_path, 'r') as file:
        line = file.readline()

        # omit trash
        while line and line[0] is not '#':
            line = file.readline()

        # read pure data
        while line:
            # if line contains title of new division
            if line[0] is '#':
                # if this is a special line - type
                if '#TYPE'.lower() in line.lower():
                    net_dict.update({'type': [line[6:-1]]})  # '6:-1' to save the name of type
                # else if it is a normal division
                else:
                    net_dict.update({name: tab})
                    name = line[1:-1].lower()  # omitting '#' and '\n'
                    tab = []
            line = file.readline()
            # don't save line with only whitespaces or if line contains a title of new division
            if not line.isspace() and '#' not in line:
                line = line.translate({ord(char): None for char in string.whitespace})
                tab.append(line.split(','))

        # append last line to dictionary
        net_dict.update({name: tab[:-1]})
    del net_dict['foo']

    return net_dict


def create_directory(dest_path):
    """
    Method checks out if given directory exists and if doesn't it just creates it

    :param dest_path: (str) absolute path to create folder
    :return: (int) 0 if directory has been created, 1 if directory had been existed before method call
    """

    if path.exists(dest_path):
        return 1
    else:
        os.mkdir(dest_path)
        return 0


'''
a = read_mlx('/Users/michal/PycharmProjects/network_diffusion/network_records/florentine.mpx')
a = read_mlx('/Users/michal/PycharmProjects/network_diffusion/network_records/test_bad')
a = read_mlx('/Users/michal/PycharmProjects/network_diffusion/network_records/fftwyt.mpx')
a = read_mlx('/Users/michal/PycharmProjects/network_diffusion/network_records/monastery.mpx')
a = read_mlx('/Users/michal/PycharmProjects/network_diffusion/requirements.txt')
print(a.keys())
for k, v in a.items():
    print(k, '\n', v)
'''
