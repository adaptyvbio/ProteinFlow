"""Metrics for evaluating prediction quality."""

import os

import Bio.PDB
import biotite.structure.io as bsio
import blosum as bl
import esm
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


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
    chain_sequences,
    predict_masks,
    esm_model_name="esm2_t30_150M_UR50D",
    esm_model_objects=None,
):
    """Compute pseudo log likelihood.

    Parameters
    ----------
    chain_sequences : list of str
        List of chain sequences (strings of amino acid codes)
    predict_masks : list of np.ndarray
        List of predict masks corresponding to the sequences (arrays of 0 and 1 where 1 indicates a predicted residue)
    esm_model_name : str, default "esm2_t30_150M_UR50D"
        Name of the ESM-2 model to use
    esm_model_objects : tuple, optional
        Tuple of ESM-2 model, batch converter and tok_to_idx dictionary (if not None, `esm_model_name` will be ignored)

    Returns
    -------
    pll: float
        Pseudo log likelihood

    """
    predict_mask = []
    for mask in predict_masks:
        predict_mask.append(mask)
        predict_mask.append(np.zeros(2))
    predict_mask = np.concatenate(predict_mask, axis=0)
    predict_idx = np.where(predict_mask)[0]
    sequence = "<eos><cls>".join(chain_sequences)

    if esm_model_objects is None:
        esm_model, batch_converter, tok_to_idx = _get_esm_model(esm_model_name)
    else:
        esm_model, batch_converter, tok_to_idx = esm_model_objects
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


def aligned_ca_rmsd(coordinates1, coordinates2):
    """Calculate CA RMSD between two structures.

    Parameters
    ----------
    coordinates1 : np.ndarray
        The CA coordinates array of the first structure, shaped `(L, 3)`
    coordinates2 : ProteinEntry
        The CA coordinates array of the second structure, shaped `(L, 3)`

    Returns
    -------
    rmsd : float
        The RMSD between the two structures

    """
    ref_atoms = []
    sample_atoms = []

    for coord in coordinates1:
        # Append CA atom to list
        atom = Bio.PDB.Atom.Atom(
            "CA",
            coord[2],
            bfactor=None,
            occupancy=1.0,
            altloc=None,
            fullname="CA",
            serial_number=None,
        )
        ref_atoms.append(atom)

    for coord in coordinates2:
        # Append CA atom to list
        atom = Bio.PDB.Atom.Atom(
            "CA",
            coord[2],
            bfactor=None,
            occupancy=1.0,
            altloc=None,
            fullname="CA",
            serial_number=None,
        )
        sample_atoms.append(atom)

    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, sample_atoms)
    return super_imposer.rms


def esmfold_generate(sequences, filepaths=None):
    """Generate PDB structures using ESMFold.

    Note that you need to install `fair-esm` with the `esmfold` option (see https://github.com/facebookresearch/esm/tree/main).
    The model also requires > 16GB CPU and GPU memory.

    Parameters
    ----------
    sequences : list of str
        List of sequences to be generated (chains separated with `':'`)
    filepaths : list of str, default None
        List of filepaths for the generated structures

    """
    assert filepaths is None or len(filepaths) == len(sequences)
    print("Loading the ESMFold model...")
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    print("Model loaded.")
    if filepaths is None:
        filepaths = [
            os.path.join("esmfold_output", f"seq_{i}.pdb")
            for i in range(len(sequences))
        ]
    with torch.no_grad():
        for sequence, path in tqdm(zip(sequences, filepaths), total=len(sequences)):
            output = model.infer_pdb(sequence)
            with open(path, "w") as f:
                f.write(output)


def esmfold_plddt(filepath, predict_mask=None):
    """Get the average PLDDT score of a structure generated by ESMFold.

    Parameters
    ----------
    filepath : str
        Filepath of the structure
    predict_mask : np.ndarray, default None
        Predict mask of the structure (1 indicates a predicted residue, 0 otherwise)

    Returns
    -------
    plddt : float
        Average PLDDT score of the structure

    """
    struct = bsio.load_structure(filepath, extra_fields=["b_factor"])
    if predict_mask is not None:
        order = -1
        order_array = []
        for i in range(len(struct)):
            if struct[i].atom_name == "N":
                order += 1
            order_array.append(order)
        b_factor = [
            atom.b_factor
            for order, atom in zip(order_array, struct)
            if predict_mask[order] == 1
        ]
        return np.array(b_factor).mean()
    else:
        return struct.b_factor.mean()
