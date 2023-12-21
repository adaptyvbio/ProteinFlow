"""Metrics for evaluating prediction quality."""

import os

import biotite.structure.io as bsio
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from proteinflow.extra import requires_extra

try:
    import blosum as bl
except ImportError:
    pass
try:
    import esm
except ImportError:
    pass
try:
    from tmtools import tm_align
except ImportError:
    pass
try:
    import ablang
except ImportError:
    pass
try:
    from igfold import IgFoldRunner
except ImportError:
    pass
try:
    from ImmuneBuilder import ABodyBuilder2, NanoBodyBuilder2, TCRBuilder2
except ImportError:
    pass


@requires_extra("blosum")
def blosum62_score(seq_before, seq_after):
    """Calculate the BLOSUM62 score between two sequences.

    Parameters
    ----------
    seq_before : str
        The sequence before the mutation
    seq_after : str
        The sequence after the mutation

    Returns
    -------
    score : int
        The BLOSUM62 score between the two sequences

    """
    assert len(seq_before) == len(seq_after)
    matrix = bl.BLOSUM(62)
    score = 0
    for x_before, x_after in zip(seq_before, seq_after):
        score += matrix[x_before][x_after]
    return score


@requires_extra("blosum")
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


@requires_extra("esm", install_name="fair-esm")
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


@requires_extra("ablang")
def ablang_pll(
    sequence,
    predict_mask,
    ablang_model_name="heavy",
    average=False,
):
    """Compute pseudo log likelihood.

    Note that you need to install `ablang` (see https://github.com/oxpig/AbLang/tree/main).

    Parameters
    ----------
    sequence : str
        Chain sequence (string of amino acid codes)
    predict_mask : np.ndarray
        Predict mask corresponding to the sequence (array of 0 and 1 where 1 indicates a predicted residue)
    ablang_model_name : {"heavy", "light"}, default "heavy"
        Name of the AbLang model to use
    average : bool, default False
        Whether to average the pseudo log likelihood over the residues

    Returns
    -------
    pll: float
        Pseudo log likelihood

    """
    ablang_model = ablang.pretrained(
        ablang_model_name
    )  # Use "light" if you are working with light chains
    ablang_model.freeze()

    sequences = []
    sequence = list(sequence)
    predict_idx = np.where(predict_mask)[0]
    for i in predict_idx:
        sequences.append("".join(sequence[:i]) + "*" + "".join(sequence[i + 1 :]))

    logits = ablang_model(sequences, mode="likelihood")[:, 1:]
    exp_logits = np.exp(logits)
    prob = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    true_idx = [
        ablang_model.tokenizer.vocab_to_token[x] - 1
        for x in np.array(sequence)[predict_idx]
    ]

    prob = prob[range(prob.shape[0]), predict_idx, true_idx]
    pll = np.log(prob).sum()
    if average:
        pll /= len(predict_idx)
    return pll


@requires_extra("esm", install_name="fair-esm")
def esm_pll(
    chain_sequences,
    predict_masks,
    esm_model_name="esm2_t30_150M_UR50D",
    esm_model_objects=None,
    average=False,
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
    average : bool, default False
        Whether to average the pseudo log likelihood over the residues

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
    sequence = []
    for i, seq in enumerate(chain_sequences):
        sequence += list(seq)
        if i != len(chain_sequences) - 1:
            sequence += ["<eos>", "<cls>"]

    if esm_model_objects is None:
        esm_model, batch_converter, tok_to_idx = _get_esm_model(esm_model_name)
    else:
        esm_model, batch_converter, tok_to_idx = esm_model_objects
    pll = 0
    for i in predict_idx:
        sequence_ = "".join(sequence[:i]) + "<mask>" + "".join(sequence[i + 1 :])
        _, _, batch_tokens = batch_converter([(0, sequence_)])
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to("cuda")
        with torch.no_grad():
            results = esm_model(batch_tokens, repr_layers=[6], return_contacts=False)
        logits = results["logits"][0, i + 1].detach().cpu()
        tok_idx = tok_to_idx[sequence[i]]
        prob = F.softmax(logits[4:24], dim=-1)[tok_idx - 4]
        pll += torch.log(prob).item()
    if average:
        pll /= len(predict_idx)
    return pll


def ca_rmsd(coordinates1, coordinates2):
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
    return np.sqrt(((coordinates1 - coordinates2) ** 2).sum(axis=-1).mean())


@requires_extra("tmtools")
def tm_score(coordinates1, coordinates2, sequence1, sequence2):
    """Calculate TM-score between two structures.

    Parameters
    ----------
    coordinates1 : np.ndarray
        The CA coordinates array of the first structure, shaped `(L, 3)`
    coordinates2 : ProteinEntry
        The CA coordinates array of the second structure, shaped `(L, 3)`
    sequence1 : str
        The sequence of the first structure
    sequence2 : str
        The sequence of the second structure

    Returns
    -------
    tm_score : float
        The TM-score between the two structures

    """
    res = tm_align(coordinates1, coordinates2, sequence1, sequence2)
    return (res.tm_norm_chain1 + res.tm_norm_chain2) / 2


@requires_extra("esm", install_name="fair-esm[esmfold]")
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
        if not os.path.exists("esmfold_output"):
            os.mkdir("esmfold_output")
        filepaths = [
            os.path.join("esmfold_output", f"seq_{i}.pdb")
            for i in range(len(sequences))
        ]
    with torch.no_grad():
        for sequence, path in tqdm(zip(sequences, filepaths), total=len(sequences)):
            output = model.infer_pdb(sequence)
            with open(path, "w") as f:
                f.write(output)


@requires_extra("igfold")
def igfold_generate(sequence_dicts, filepaths=None, use_openmm=False):
    """Generate PDB structures using IgFold.

    Note that you need to install `igfold` (see https://github.com/Graylab/IgFold).

    Parameters
    ----------
    sequence_dicts : list of dict
        List of sequence dictionaries (keys: "H", "L" for heavy and light chains)
    filepaths : list of str, optional
        List of filepaths for the generated structures
    use_openmm : bool, default False
        Whether to use refinement with OpenMM

    """
    assert filepaths is None or len(filepaths) == len(sequence_dicts)
    igfold = IgFoldRunner()
    folder = "igfold_refine_output" if use_openmm else "igfold_output"
    if filepaths is None:
        if not os.path.exists(folder):
            os.mkdir(folder)
        filepaths = [
            os.path.join(folder, f"seq_{i}.pdb") for i in range(len(sequence_dicts))
        ]
    for seqs, path in tqdm(zip(sequence_dicts, filepaths), total=len(sequence_dicts)):
        igfold.fold(
            path,  # Output PDB file
            sequences=seqs,  # Antibody sequences
            do_refine=use_openmm,  # Refine the antibody structure
            use_openmm=use_openmm,  # Use OpenMM for refinement
            do_renum=False,  # Renumber predicted antibody structure (Chothia)
        )


@requires_extra("ImmuneBuilder")
def immunebuilder_generate(sequence_dicts, filepaths=None, protein_type="antibody"):
    """Generate PDB structures using ImmuneBuilder.

    Note that you need to install `immunebuilder` (see https://github.com/oxpig/ImmuneBuilder)

    Parameters
    ----------
    sequence_dicts : list of dict
        List of sequence dictionaries (keys: "H", "L" for heavy and light chains)
    filepaths : list of str, optional
        List of filepaths for the generated structures
    protein_type: {"antibody", "nanobody", "tcr"}
        Type of the structure to generate

    """
    predictor_classes = {
        "antibody": ABodyBuilder2,
        "nanobody": NanoBodyBuilder2,
        "tcr": TCRBuilder2,
    }
    predictor = predictor_classes[protein_type]()
    folder = "immunebuilder_output"
    if filepaths is None:
        if not os.path.exists(folder):
            os.mkdir(folder)
        filepaths = [
            os.path.join(folder, f"seq_{i}.pdb") for i in range(len(sequence_dicts))
        ]
    for seqs, path in tqdm(zip(sequence_dicts, filepaths), total=len(sequence_dicts)):
        out = predictor.predict(seqs)
        out.save(path)


def confidence_from_file(filepath, predict_mask=None):
    """Get the average pLDDT or pRMSD score of a structure generated with ESMFold or IgFold.

    This function loads the metric that is stored in the B-factor column of the PDB file.
    For files generated with ESMFold, the metric is pLDDT; for IgFold and ImmuneBuilder, the metric is pRMSD.

    Parameters
    ----------
    filepath : str
        Filepath of the structure
    predict_mask : np.ndarray, default None
        Predict mask of the structure (1 indicates a predicted residue, 0 otherwise)

    Returns
    -------
    confidence : float
        Average PLDDT / pRMSD score of the structure

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
