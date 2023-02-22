import os
import subprocess
import editdistance
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import Counter


def _open_pdb(file):
    """
    Open a PDB file in the pickle format that follows the dwnloading and processing of the database
    """

    with open(file, "rb") as f:
        return pkl.load(f)


def _compare_identity(seq, seqs, threshold):
    """
    Assess whether a sequence is in a list of sequences (in the sense that it shares at least 90% to one of the sequences in the list)
    """

    for s in seqs:
        if editdistance.eval(s, seq) / max(len(s), len(seq)) <= (1 - threshold):
            return True

    return False


def _compare_seqs(seqs1, seqs2, threshold):
    """
    Assess whether 2 lists of sequences contain exactly the same set of sequences
    """

    for seq in seqs1:
        if not _compare_identity(seq, seqs1, threshold):
            return False

    for seq in seqs2:
        if not _compare_identity(seq, seqs2, threshold):
            return False

    return True


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


def _remove_database_redundancies(dir, seq_identity_threshold=0.9):
    """
    Remove all biounits in the database that are copies to another biounits in terms of sequence

    Sequence identity is definded by the 'seq_identity_threshold' parameter for robust detection of sequence similarity (missing residues, point mutations, ...).

    Parameters
    ----------
    dir : str
        the path to the database where all the biounits are stored in pickle files after their processing
    seq_identity_threshold : float, default .9
        the threshold that determines up to what percentage identity sequences are considered as the same

    Returns
    -------
    total_removed : int
        the total number of removed biounits
    """

    all_files = np.array(os.listdir(dir))
    all_pdbs = np.array([file[:4] for file in all_files])
    pdb_counts = Counter(all_pdbs)
    pdbs_to_check = [pdb for pdb in pdb_counts.keys() if pdb_counts[pdb] > 1]
    total_removed = []

    for pdb in tqdm(pdbs_to_check):
        biounits_list = np.array(
            [os.path.join(dir, file) for file in all_files[all_pdbs == pdb]]
        )
        biounits_list = sorted(biounits_list)
        redundancies = _check_biounits(biounits_list, seq_identity_threshold)
        if redundancies != []:
            for k in redundancies:
                total_removed.append(os.path.basename(biounits_list[k]).split(".")[0])
                subprocess.run(["rm", biounits_list[k]])

    return total_removed
