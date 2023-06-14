import os
import subprocess
import editdistance
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import Counter
from proteinflow.pdb import _open_pdb

from proteinflow.sequences import _compare_seqs


def _check_biounits(biounits_list, threshold):
    """
    Return the indexes of the redundant biounits within the list of files given by `biounits_list`
    """

    biounits = [_open_pdb(b) for b in biounits_list]
    indexes = []

    for k, b1 in enumerate(biounits):
        if k not in indexes:
            b1_seqs = [b1[chain]["seq"] for chain in b1.keys()]
            for l, b2 in enumerate(biounits[k + 1 :]):
                if len(b1.keys()) != len(b2.keys()):
                    continue

                b2_seqs = [b2[chain]["seq"] for chain in b2.keys()]
                if _compare_seqs(b1_seqs, b2_seqs, threshold):
                    indexes.append(k + l + 1)

    return indexes

