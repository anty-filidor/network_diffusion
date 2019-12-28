import string
import os.path
from os import path


def read_mlx(file_path):
    """
    This function reads multilayer network from mlx file. Note that for now API allows only to create a multiplex net.
    Thus some parts of file are un unuseful, but they are returned to maintain further scalability of this project.

    **A mlx file format**
        A mlx file is text file which is divided into several parts:
            * type - a type of network e.g. *multiplex*
            * layers - list of layer names followed by its types e.g. *layer1,UNDIRECTED*
            * actor (nodes) attributes - list of actor attributes followed by its type e.g. *attribute1, NUMERIC*
            * actors (nodes) - list of actors followed by its attributes values e.g. *actor1,53,2,10*
            * edge attributes - list of edge attributes e.g. *layer1,layer2,NUMERIC*
            * edges - list of edges in form *actor1,actor2,layer1*
        Each part starts with special sign '#'. Between parts there is often an empty line. Sequence of given elements
        does not matter for function. Note, that for human readability of file it does. Below there is an example of mlx
        file ::

        #TYPE multiplex

        #LAYERS
        marriage,UNDIRECTED
        business,UNDIRECTED

        #ACTOR ATTRIBUTES
        priorates, NUMERIC
        wealth, NUMERIC

        #ACTORS
        Acciaiuoli,53,2
        Albizzi,65,3
        Barbadori,0,14


    :param file_path: path to file
    :return: a dictionary with net to create class
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
print(a.keys(), '\n', a)
'''
