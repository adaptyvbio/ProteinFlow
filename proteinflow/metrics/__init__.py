"""Metrics for evaluating prediction quality."""

import blosum as bl
import numpy as np


def blosum62_score(seq_before, seq_after):
    """Calculate the BLOSUM62 score between two sequences.

    Parameters
    ----------
    seq_before : str
        The sequence before the mutation.
    seq_after : str
        The sequence after the mutation.

    Returns
    -------
    score : int
        The BLOSUM62 score between the two sequences.

    """
    assert len(seq_before) == len(seq_after)
    matrix = bl.BLOSUM(62)
    score = 0
    for x_before, x_after in zip(seq_before, seq_after):
        score += matrix[x_before][x_after]
    return score


def long_repeat_num(seq, thr=5):
    """Calculate the number of long repeats in a sequence.

    Parameters
    ----------
    seq : str
        The sequence to be evaluated.
    thr : int, default 5
        The threshold of the length of a repeat.

    Returns
    -------
    num : int
        The number of long repeats in the sequence.

    """
    arr = np.array(list(seq))
    # Find the indices where the array changes its value
    changes = np.flatnonzero(arr[:-1] != arr[1:])
    # Split the array into consecutive groups
    groups = np.split(arr, changes + 1)
    # Filter groups that are longer than N
    long_groups = filter(lambda g: len(g) > thr, groups)
    # Count the number of long groups
    count = sum(1 for _ in long_groups)
    return count
