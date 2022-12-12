from helpers import load_obj, save_obj
from os import getcwd, mkdir
from os.path import join
from time import strftime
import argparse
import sys


def generate_data_files(filename, path_to_file='', batch_size=1000, max_num_batches=100):
    """
    Generate a set of files in a new directory, where each file is a list of graphs, with the number of graphs in that
    class
    :param filename:
    :param path_to_file:
    :param batch_size:
    :param max_num_batches:
    :return: number of files generated
    """
    # Load full data set
    full_data_set = load_obj(filename, path_to_file)

    # Create new directory for output files
    timestr = strftime("%d_%m_%Y-%H_%M")
    cwd = getcwd()
    name = "graph_batch_files" + timestr
    print(name)
    path = join(cwd, name)
    mkdir(path)
    tot_num_data = len(full_data_set)
    a = tot_num_data // batch_size
    if tot_num_data % batch_size == 0:
        n_batches_reqd = a
    else:
        n_batches_reqd = a + 1
    if n_batches_reqd > max_num_batches:
        n_batches = max_num_batches
        last_batch_full = True
    else:
        n_batches = n_batches_reqd
        last_batch_full = False

    for i in range(0, n_batches):
        if i < n_batches - 1:
            batch = full_data_set[i * batch_size:(i + 1) * batch_size]
        else:
            if last_batch_full:
                batch = full_data_set[i * batch_size:(i + 1) * batch_size]
            else:
                batch = full_data_set[(n_batches - 1) * batch_size:]
        save_obj(batch, f'graph_data_batch{i}', path)
    return n_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pass variables for Loss tolerance SPF")
    parser.add_argument(
        "-fn",
        "--filename",
        help="name of graph data",
        type=str,
        default="",
    )
    parser.add_argument(
        "-p",
        "--path_to_data",
        help="path to the directory containing the graph data",
        type=str,
        default="",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="size of batches to examine",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "-mb",
        "--max_batches",
        help="maximum number of datafiles to generate",
        type=int,
        default=100,
    )

    arguments = parser.parse_args(sys.argv[1:])
    filename = arguments.filename
    path_to_data = arguments.path_to_data
    batch_size = arguments.batch_size
    max_batches = arguments.max_batches
    # print(filename, batch_size)

    generate_data_files(filename=filename, path_to_file=path_to_data, batch_size=batch_size, max_num_batches=max_batches)
