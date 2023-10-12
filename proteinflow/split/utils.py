"""Splitting util functions."""

import copy
import os
import pickle
from collections import defaultdict

import editdistance
import numpy as np
from tqdm import tqdm


def _unique_chains(seqs_list):
    """Get unique chains."""
    new_seqs_list = [seqs_list[0]]
    chains = [new_seqs_list[0][0]]

    for seq in seqs_list[1:]:
        if seq[0] not in chains:
            new_seqs_list.append(seq)
            chains.append(seq[0])

    return new_seqs_list


def _merge_chains(seqs_dict_):
    """Look into the chains of each PDB and regroup redundancies (at 90% sequence identity)."""
    seqs_dict = copy.deepcopy(seqs_dict_)
    pdbs_to_delete = []

    for pdb in tqdm(seqs_dict.keys()):
        if seqs_dict[pdb] == []:
            pdbs_to_delete.append(pdb)
            continue

        seqs_dict[pdb] = _unique_chains(seqs_dict[pdb])
        groups, ref_seqs, indexes = [], [], []

        for k in range(len(seqs_dict[pdb])):
            if k in indexes:
                continue
            group = [seqs_dict[pdb][k][0]]
            ref_seq = seqs_dict[pdb][k][1]
            ref_seqs.append(ref_seq)
            indexes.append(k)

            for i in range(k + 1, len(seqs_dict[pdb])):
                chain, seq = seqs_dict[pdb][i][0], seqs_dict[pdb][i][1]
                if (
                    i in indexes
                    or len(seq) > 1.1 * len(ref_seq)
                    or len(seq) < 0.9 * len(ref_seq)
                    or editdistance.eval(seq, ref_seq) / max(len(seq), len(ref_seq))
                    > 0.1
                ):
                    continue
                group.append(chain)
                indexes.append(i)

            groups.append(group)

        new_group = []
        for group, seq in zip(groups, ref_seqs):
            new_group.append(("-".join(group), seq))
        seqs_dict[pdb] = new_group

    for pdb in pdbs_to_delete:
        del seqs_dict[pdb]

    return seqs_dict


def _load_pdbs(dir, cdr=None):
    """Load biounits and group their sequences by PDB and similarity (90%)."""
    seqs_dict = defaultdict(lambda: [])

    for file in tqdm([x for x in os.listdir(dir) if x.endswith(".pickle")]):
        load_path = os.path.join(dir, file)
        if os.path.isdir(load_path):
            continue
        with open(load_path, "rb") as f:
            pdb_dict = pickle.load(f)
        if cdr is None:
            seqs = [(chain, pdb_dict[chain]["seq"]) for chain in pdb_dict.keys()]
        else:
            seqs = [
                (
                    chain,
                    "".join(
                        np.array(list(pdb_dict[chain]["seq"]))[
                            pdb_dict[chain]["cdr"] == cdr
                        ].tolist()
                    ),
                )
                for chain in pdb_dict.keys()
            ]
            seqs = [(chain, seq) for chain, seq in seqs if len(seq) > 0]
        seqs_dict[file[:4]] += seqs

    return seqs_dict


def _write_fasta(fasta_path, merged_seqs_dict):
    """Write a fasta file containing all the sequences contained in the merged_seqs_dict dictionary."""
    with open(fasta_path, "w") as f:
        for k in merged_seqs_dict.keys():
            for chain, seq in merged_seqs_dict[k]:
                f.write(">" + k + "_" + chain + "\n")
                f.write(seq + "\n")


def _retrieve_seqs_names_list(merged_seqs_dict):
    """Retrieve all the sequences names that are contained in the merged_seqs_dict (same names as in the fasta file in input to MMSeqs2)."""
    seqs_names = []
    for k in merged_seqs_dict.keys():
        for chain, _ in merged_seqs_dict[k]:
            seqs_names.append(k + "_" + chain)

    return seqs_names


def _create_pdb_seqs_dict(seqs_names_list):
    """Return a dictionary that has the PDB ids as keys and the list of all the chain names that correspond to this PDB."""
    pdb_seqs_dict = defaultdict(lambda: [])
    for name in seqs_names_list:
        pdb_seqs_dict[name[:4]].append(name[5:])

    return pdb_seqs_dict


def _test_availability(
    size_array,
    n_samples,
):
    """Test if there are enough groups in each class to construct a dataset with the required number of samples."""
    present = np.sum(size_array != 0, axis=0)
    return present[0] > n_samples, present[1] > n_samples, present[2] > n_samples


def _find_correspondences(files, dataset_dir):
    """Return a dictionary that contains all the biounits in the database (keys) and the list of all the chains that are in these biounits (values)."""
    correspondences = defaultdict(lambda: [])
    for file in files:
        biounit = file
        with open(os.path.join(dataset_dir, file), "rb") as f:
            keys = pickle.load(f)
            for k in keys:
                correspondences[biounit].append(k)

    return correspondences


def _biounits_in_clusters_dict(clusters_dict, excluded_files=None):
    """Return the list of all biounit files present in clusters_dict."""
    if len(clusters_dict) == 0:
        return np.array([])
    if excluded_files is None:
        excluded_files = []
    return np.unique(
        [
            c[0]
            for c in list(np.concatenate(list(clusters_dict.values())))
            if c[0] not in excluded_files
        ]
    )


def _exclude(clusters_dict, set_to_exclude, exclude_based_on_cdr=None):
    """Exclude biounits from clusters_dict.

    Parameters
    ----------
    clusters_dict : dict
        dictionary of clusters
    set_to_exclude : set
        set of biounits to exclude
    exclude_based_on_cdr : str, default None
        if not None, exclude based only on clusters based on this CDR (e.g. "H3")

    """
    excluded_set = set()
    excluded_dict = defaultdict(set)
    for cluster in list(clusters_dict.keys()):
        files = clusters_dict[cluster]
        exclude = False
        for biounit in files:
            if biounit[0] in set_to_exclude:
                exclude = True
                break
        if exclude:
            if exclude_based_on_cdr is not None:
                if not cluster.endswith(exclude_based_on_cdr):
                    continue
            for biounit in files:
                clusters_dict[cluster].remove(biounit)
                excluded_dict[cluster].add(biounit)
                excluded_set.add(biounit)
    return excluded_dict, excluded_set
