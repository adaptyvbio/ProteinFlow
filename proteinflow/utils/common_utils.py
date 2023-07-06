import itertools
import os
import pickle
import subprocess
from collections import defaultdict

import numpy as np


class PDBError(ValueError):
    pass


def _split_every(n, iterable):
    """Split iterable into chunks. From https://stackoverflow.com/a/1915307/2780645."""
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))


def _clean(pdb_id, tmp_folder):
    """
    Remove all temporary files associated with a PDB ID
    """
    for file in os.listdir(tmp_folder):
        if file.startswith(f"{pdb_id}."):
            subprocess.run(
                ["rm", os.path.join(tmp_folder, file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _raise_rcsbsearch(e):
    """
    Raise a RuntimeError if the error is due to rcsbsearch
    """

    if "404 Client Error" in str(e):
        raise RuntimeError(
            'Querying rcsbsearch is failing. Please install a version of rcsbsearch where this error is solved:\npython -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"'
        )
    else:
        raise e


def _make_sabdab_html(method, resolution_thr):
    """
    Make a URL for SAbDab search
    """

    html = f"https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?ABtype=All&method={'+'.join(method)}&species=All&resolution={resolution_thr}&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=All&isin_covabdab=All&isin_therasabdab=All&chothiapos=&restype=ALA&field_0=Antigens&keyword_0=#downloads"
    return html


def _test_availability(
    size_array,
    n_samples,
):
    """
    Test if there are enough groups in each class to construct a dataset with the required number of samples
    """

    present = np.sum(size_array != 0, axis=0)
    return present[0] > n_samples, present[1] > n_samples, present[2] > n_samples


def _find_correspondences(files, dataset_dir):
    """
    Return a dictionary that contains all the biounits in the database (keys) and the list of all the chains that are in these biounits (values)
    """

    correspondences = defaultdict(lambda: [])
    for file in files:
        biounit = file
        with open(os.path.join(dataset_dir, file), "rb") as f:
            keys = pickle.load(f)
            for k in keys:
                correspondences[biounit].append(k)

    return correspondences
