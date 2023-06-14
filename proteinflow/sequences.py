import editdistance
import numpy as np

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

    with open(fasta_file, "r") as f:
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
