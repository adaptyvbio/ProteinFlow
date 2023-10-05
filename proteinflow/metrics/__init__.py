"""Metrics for evaluating prediction quality."""

import blosum as bl
import esm
import numpy as np
import torch
from torch.nn import functional as F


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


def _get_esm_model(esm_model_name):
    """Get ESM model, batch converter and tok_to_idx dictionary."""
    model_dict = {
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
    }
    esm_model, alphabet = model_dict[esm_model_name]()
    if torch.cuda.is_available():
        esm_model.to("cuda")
    batch_converter = alphabet.get_batch_converter()
    tok_to_idx = alphabet.tok_to_idx
    return esm_model, batch_converter, tok_to_idx


def esm_pll(
    entry=None,
    chain_sequences=None,
    predict_masks=None,
    esm_model_name="esm2_t30_150M_UR50D",
):
    """Compute pseudo log likelihood.

    The input is either a `ProteinEntry` object or lists of chain sequences and predict masks.
    If both are provided, the `ProteinEntry` object is used.

    Parameters
    ----------
    entry : ProteinEntry, optional
        ProteinEntry object (with `predict_mask` not `None`)
    chain_sequences : list of str, optional
        List of chain sequences (strings of amino acid codes)
    predict_masks : list of np.ndarray, optional
        List of predict masks (arrays of 0 and 1 where 1 indicates a masked residue)
    esm_model_name : str, default "esm2_t30_150M_UR50D"
        Name of the ESM-2 model to use

    Returns
    -------
    pll: float
        Pseudo log likelihood

    """
    assert entry is not None or (
        chain_sequences is not None and predict_masks is not None
    )
    if entry is not None:
        chains = entry.get_chains()
        chain_sequences = [entry.get_sequence(chains=[chain]) for chain in chains]
        predict_masks = [
            (entry.get_predict_mask(chains=[chain])).astype(float) for chain in chains
        ]
    predict_mask = []
    for mask in predict_masks:
        predict_mask.append(mask)
        predict_mask.append(np.zeros(2))
    predict_mask = np.concatenate(predict_mask, axis=0)
    predict_idx = np.where(predict_mask)[0]
    sequence = "<eos><cls>".join(chain_sequences)

    esm_model, batch_converter, tok_to_idx = _get_esm_model(esm_model_name)
    pll = 0
    for i in predict_idx:
        sequence_ = sequence[:i] + "<mask>" + sequence[i + 1 :]
        _, _, batch_tokens = batch_converter([(0, sequence_)])
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to("cuda")
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
        logits = results["logits"][0, i].detach().cpu()
        tok_idx = tok_to_idx[sequence[i]]
        prob = F.softmax(logits[4:24], dim=-1)[tok_idx - 4]
        pll += torch.log(prob)
    return pll
