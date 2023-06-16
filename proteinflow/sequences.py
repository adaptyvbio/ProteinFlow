import copy
import os
import pickle
from collections import defaultdict

import editdistance
import numpy as np
from tqdm import tqdm

from proteinflow.constants import CDR_VALUES


def _retrieve_author_chain(chain):
    """
    Retrieve the (author) chain names present in the chain section (delimited by '|' chars) of a header line in a fasta file
    """

    if "auth" in chain:
        return chain.split(" ")[-1][:-1]

    return chain


def _retrieve_chain_names(entry):
    """
    Retrieve the (author) chain names present in one header line of a fasta file (line that begins with '>')
    """

    entry = entry.split("|")[1]

    if "Chains" in entry:
        return [_retrieve_author_chain(e) for e in entry[7:].split(", ")]

    return [_retrieve_author_chain(entry[6:])]


def _retrieve_fasta_chains(fasta_file):
    """
    Return a dictionary containing all the (author) chains in a fasta file (keys) and their corresponding sequence
    """

    with open(fasta_file) as f:
        lines = np.array(f.readlines())

    indexes = np.array([k for k, l in enumerate(lines) if l[0] == ">"])
    starts = indexes + 1
    ends = list(indexes[1:]) + [len(lines)]
    names = lines[indexes]
    seqs = ["".join(lines[s:e]).replace("\n", "") for s, e in zip(starts, ends)]

    out_dict = {}
    for name, seq in zip(names, seqs):
        for chain in _retrieve_chain_names(name):
            out_dict[chain] = seq

    return out_dict


def _get_chothia_cdr(num_array, chain_type):
    arr = [CDR_VALUES[chain_type][int(x.split("_")[0])] for x in num_array]
    return np.array(arr)


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
        if not _compare_identity(seq, seqs2, threshold):
            return False

    for seq in seqs2:
        if not _compare_identity(seq, seqs1, threshold):
            return False

    return True


def _unique_chains(seqs_list):
    """
    Get unique chains
    """

    new_seqs_list = [seqs_list[0]]
    chains = [new_seqs_list[0][0]]

    for seq in seqs_list[1:]:
        if seq[0] not in chains:
            new_seqs_list.append(seq)
            chains.append(seq[0])

    return new_seqs_list


def _merge_chains(seqs_dict_):
    """
    Look into the chains of each PDB and regroup redundancies (at 90% sequence identity)
    """

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
    """
    Load biounits and group their sequences by PDB and similarity (90%)
    """

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
    """
    Write a fasta file containing all the sequences contained in the merged_seqs_dict dictionary
    """

    with open(fasta_path, "w") as f:
        for k in merged_seqs_dict.keys():
            for chain, seq in merged_seqs_dict[k]:
                f.write(">" + k + "_" + chain + "\n")
                f.write(seq + "\n")


def _retrieve_seqs_names_list(merged_seqs_dict):
    """
    Retrieve all the sequences names that are contained in the merged_seqs_dict (same names as in the fasta file in input to MMSeqs2)
    """

    seqs_names = []
    for k in merged_seqs_dict.keys():
        for chain, _ in merged_seqs_dict[k]:
            seqs_names.append(k + "_" + chain)

    return seqs_names


def _create_pdb_seqs_dict(seqs_names_list):
    """
    Return a dictionary that has the PDB ids as keys and the list of all the chain names that correspond to this PDB
    """

    pdb_seqs_dict = defaultdict(lambda: [])
    for name in seqs_names_list:
        pdb_seqs_dict[name[:4]].append(name[5:])

    return pdb_seqs_dict
