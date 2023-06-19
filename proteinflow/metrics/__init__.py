"""Metrics for evaluating prediction quality."""

import blosum as bl


def blosum62(seq_before, seq_after):
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
