"""
`bestprot` is a very cool library.

## Processing pipeline
...

## Pre-computed data
...

## Handling data
...
"""

__pdoc__ = {"utils": False, "scripts": False}
__docformat__ = "numpy"

from bestprot.utils.filter_database import _remove_database_redundancies
from bestprot.utils.process_pdb import (
    _align_pdb,
    _open_pdb,
    PDBError,
    _get_pdb_file,
    _s3list,
    SIDECHAIN_ORDER,
)
from bestprot.utils.cluster_and_partition import _build_dataset_partition, _check_mmseqs
from bestprot.utils.split_dataset import _download_dataset, _split_data

import warnings
import os
import pickle
from collections import defaultdict
from rcsbsearch import Attr
from datetime import datetime
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import pickle
from p_tqdm import p_map
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
from numpy import linalg
from einops import rearrange
import boto3


MAIN_ATOMS = {
    "GLY": None,
    "ALA": 0,
    "VAL": 0,
    "LEU": 1,
    "ILE": 1,
    "MET": 2,
    "PRO": 1,
    "TRP": 5,
    "PHE": 6,
    "TYR": 7,
    "CYS": 1,
    "SER": 1,
    "THR": 1,
    "ASN": 1,
    "GLN": 2,
    "HIS": 2,
    "LYS": 3,
    "ARG": 4,
    "ASP": 1,
    "GLU": 2,
}
D3TO1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}
ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"

FEATURES_DICT = defaultdict(lambda: defaultdict(lambda: 0))
FEATURES_DICT["hydropathy"].update(
    {
        "-": 0,
        "I": 4.5,
        "V": 4.2,
        "L": 3.8,
        "F": 2.8,
        "C": 2.5,
        "M": 1.9,
        "A": 1.8,
        "W": -0.9,
        "G": -0.4,
        "T": -0.7,
        "S": -0.8,
        "Y": -1.3,
        "P": -1.6,
        "H": -3.2,
        "N": -3.5,
        "D": -3.5,
        "Q": -3.5,
        "E": -3.5,
        "K": -3.9,
        "R": -4.5,
    }
)
FEATURES_DICT["volume"].update(
    {
        "-": 0,
        "G": 60.1,
        "A": 88.6,
        "S": 89.0,
        "C": 108.5,
        "D": 111.1,
        "P": 112.7,
        "N": 114.1,
        "T": 116.1,
        "E": 138.4,
        "V": 140.0,
        "Q": 143.8,
        "H": 153.2,
        "M": 162.9,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "R": 173.4,
        "F": 189.9,
        "Y": 193.6,
        "W": 227.8,
    }
)
FEATURES_DICT["charge"].update(
    {
        **{"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1},
        **{x: 0 for x in "ABCFGIJLMNOPQSTUVWXYZ-"},
    }
)
FEATURES_DICT["polarity"].update(
    {**{x: 1 for x in "RNDQEHKSTY"}, **{x: 0 for x in "ACGILMFPWV-"}}
)
FEATURES_DICT["acceptor"].update(
    {**{x: 1 for x in "DENQHSTY"}, **{x: 0 for x in "RKWACGILMFPV-"}}
)
FEATURES_DICT["donor"].update(
    {**{x: 1 for x in "RKWNQHSTY"}, **{x: 0 for x in "DEACGILMFPV-"}}
)
_PMAP = lambda x: [
    FEATURES_DICT["hydropathy"][x] / 5,
    FEATURES_DICT["volume"][x] / 200,
    FEATURES_DICT["charge"][x],
    FEATURES_DICT["polarity"][x],
    FEATURES_DICT["acceptor"][x],
    FEATURES_DICT["donor"][x],
]


def _clean(pdb_id, tmp_folder):
    """
    Remove all temporary files associated with a PDB ID
    """
    for file in os.listdir(tmp_folder):
        if file.startswith(f"{pdb_id}."):
            subprocess.run(
                ["rm", os.path.join(tmp_folder, file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _log_exception(exception, log_file, pdb_id, tmp_folder):
    """
    Record the error in the log file
    """

    _clean(pdb_id, tmp_folder)
    if isinstance(exception, PDBError):
        with open(log_file, "a") as f:
            f.write(f"<<< {str(exception)}: {pdb_id} \n")
    else:
        with open(log_file, "a") as f:
            f.write(f"<<< Unknown: {pdb_id} \n")
            f.write(str(exception))
            f.write("\n")


def _log_removed(removed, log_file):
    """
    Record which files we removed due to redundancy
    """

    for pdb_id in removed:
        with open(log_file, "a") as f:
            f.write(f"<<< Removed due to redundancy: {pdb_id} \n")


def _get_split_dictionaries(
    tmp_folder="./data/tmp_pdb",
    output_folder="./data/pdb",
    split_tolerance=0.2,
    test_split=0.05,
    valid_split=0.05,
    out_split_dict_folder="./data/dataset_splits_dict",
):
    """
    Split preprocessed data into training, validation and test

    Parameters
    ----------
    tmp_folder : str, default "./data/tmp_pdb"
        The folder where temporary files will be saved
    output_folder : str, default "./data/pdb"
        The folder where the output files will be saved
    split_tolerance : float, default 0.2
        The tolerance on the split ratio (default 20%)
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)
    out_split_dict_folder : str, default "./data/dataset_splits_dict"
        The folder where the dictionaries containing the train/validation/test splits information will be saved"
    """

    os.makedirs(out_split_dict_folder, exist_ok=True)
    (
        train_clusters_dict,
        train_classes_dict,
        valid_clusters_dict,
        valid_classes_dict,
        test_clusters_dict,
        test_classes_dict,
    ) = _build_dataset_partition(
        output_folder,
        tmp_folder,
        valid_split=valid_split,
        test_split=test_split,
        tolerance=split_tolerance,
    )
    with open(os.path.join(out_split_dict_folder, "train.pickle"), "wb") as f:
        pickle.dump(train_clusters_dict, f)
        pickle.dump(train_classes_dict, f)
    with open(os.path.join(out_split_dict_folder, "valid.pickle"), "wb") as f:
        pickle.dump(valid_clusters_dict, f)
        pickle.dump(valid_classes_dict, f)
    with open(os.path.join(out_split_dict_folder, "test.pickle"), "wb") as f:
        pickle.dump(test_clusters_dict, f)
        pickle.dump(test_classes_dict, f)


def _run_processing(
    tmp_folder="./data/tmp_pdb",
    output_folder="./data/pdb",
    log_folder="./data/logs",
    min_length=30,
    max_length=10000,
    resolution_thr=3.5,
    missing_ends_thr=0.3,
    missing_middle_thr=0.1,
    filter_methods=True,
    remove_redundancies=False,
    seq_identity_threshold=0.9,
    n=None,
    force=False,
    tag=None,
    pdb_snapshot=None,
):
    """
    Download and parse PDB files that meet filtering criteria

    The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are
    the following:

    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order, check `sidechain_order()`),
    - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
        zeros to missing values,
    - `'seq'`: a string of length `L` with residue types.

    All errors including reasons for filtering a file out are logged in a log file.

    Parameters
    ----------
    tmp_folder : str, default "./data/tmp_pdb"
        The folder where temporary files will be saved
    output_folder : str, default "./data/pdb"
        The folder where the output files will be saved
    log_folder : str, default "./data/logs"
        The folder where the log file will be saved
    min_length : int, default 30
        The minimum number of non-missing residues per chain
    max_length : int, default 10000
        The maximum number of residues per chain (set None for no threshold)
    resolution_thr : float, default 3.5
        The maximum resolution
    missing_ends_thr : float, default 0.3
        The maximum fraction of missing residues at the ends
    missing_middle_thr : float, default 0.1
        The maximum fraction of missing residues in the middle (after missing ends are disregarded)
    filter_methods : bool, default True
        If `True`, only files obtained with X-ray or EM will be processed
    remove_redundancies : bool, default False
        If 'True', removes biounits that are doubles of others sequence wise
    seq_identity_threshold : float, default 0.9
        The threshold upon which sequences are considered as one and the same (default: 90%)
    n : int, default None
        The number of files to process (for debugging purposes)
    force : bool, default False
        When `True`, rewrite the files if they already exist
    split_tolerance : float, default 0.2
        The tolerance on the split ratio (default 20%)
    split_database : bool, default False
        Whether or not to split the database
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)
    out_split_dict_folder : str, default "./data/dataset_splits_dict"
        The folder where the dictionaries containing the train/validation/test splits information will be saved"
    tag : str, optional
        A tag to add to the log file
    pdb_snapshot : str, optional
        the PDB snapshot to use

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors
    """

    TMP_FOLDER = tmp_folder
    OUTPUT_FOLDER = output_folder
    PDB_PREFIX = "pub/pdb/data/biounit/PDB/all/"
    MIN_LENGTH = min_length
    MAX_LENGTH = max_length
    RESOLUTION_THR = resolution_thr
    MISSING_ENDS_THR = missing_ends_thr
    MISSING_MIDDLE_THR = missing_middle_thr

    if not os.path.exists(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    i = 0
    while os.path.exists(os.path.join(log_folder, f"log_{i}.txt")):
        i += 1
    LOG_FILE = os.path.join(log_folder, f"log_{i}.txt")
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n\n"
    with open(LOG_FILE, "a") as f:
        f.write(date_time)
        if tag is not None:
            f.write(f"tag: {tag}")

    # get filtered PDB ids fro PDB API
    pdb_ids = (
        Attr("rcsb_entry_info.selected_polymer_entity_types")
        .__eq__("Protein (only)")
        .or_("rcsb_entry_info.polymer_composition")
        .__eq__("protein/oligosaccharide")
    )
    # if include_na:
    #     pdb_ids = pdb_ids.or_('rcsb_entry_info.polymer_composition').in_(["protein/NA", "protein/NA/oligosaccharide"])

    pdb_ids = pdb_ids.and_("rcsb_entry_info.resolution_combined").__le__(RESOLUTION_THR)
    if filter_methods:
        pdb_ids = pdb_ids.and_("exptl.method").in_(
            ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
        )
    pdb_ids = pdb_ids.exec("assembly")
    if n is not None:
        pdbs = []
        for i, x in enumerate(pdb_ids):
            pdbs.append(x)
            if i == n:
                break
        pdb_ids = pdbs

    ordered_folders = [
        x.key + PDB_PREFIX
        for x in _s3list(
            boto3.resource("s3").Bucket("pdbsnapshots"),
            "",
            recursive=False,
            list_objs=False,
        )
    ]
    ordered_folders = sorted(
        ordered_folders, reverse=True
    )  # a list of PDB snapshots from newest to oldest
    if pdb_snapshot is not None:
        if pdb_snapshot not in ordered_folders:
            raise ValueError(
                f"The {pdb_snapshot} PDB snapshot not found, please choose from {ordered_folders}"
            )
        ind = ordered_folders.index(pdb_snapshot)
        ordered_folders = ordered_folders[ind:]

    def process_f(pdb_id, show_error=False, force=True):
        try:
            pdb_id = pdb_id.lower()
            id, biounit = pdb_id.split("-")
            target_file = os.path.join(OUTPUT_FOLDER, pdb_id + ".pickle")
            if not force and os.path.exists(target_file):
                raise PDBError("File already exists")
            pdb_file = f"{id}.pdb{biounit}.gz"
            # download
            local_path = _get_pdb_file(
                pdb_file,
                boto3.resource("s3").Bucket("pdbsnapshots"),
                tmp_folder=TMP_FOLDER,
                folders=ordered_folders,
            )
            # parse
            pdb_dict = _open_pdb(
                local_path,
                tmp_folder=TMP_FOLDER,
            )
            # filter and convert
            pdb_dict = _align_pdb(
                pdb_dict,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                max_missing_ends=MISSING_ENDS_THR,
                max_missing_middle=MISSING_MIDDLE_THR,
            )
            # save
            if pdb_dict is not None:
                with open(target_file, "wb") as f:
                    pickle.dump(pdb_dict, f)
        except Exception as e:
            if show_error:
                raise e
            else:
                _log_exception(e, LOG_FILE, pdb_id, TMP_FOLDER)

    # process_f("1a1q-3", show_error=True, force=force)

    _ = p_map(lambda x: process_f(x, force=force), pdb_ids)

    stats = get_error_summary(LOG_FILE, verbose=False)
    while "<<< PDB file not found" in stats:
        with open(LOG_FILE, "r") as f:
            lines = [
                x for x in f.readlines() if not x.startswith("<<< PDB file not found")
            ]
        os.remove(LOG_FILE)
        with open(f"{LOG_FILE}_tmp", "a") as f:
            for line in lines:
                f.write(line)
        _ = p_map(lambda x: process_f(x, force=force), stats["<<< PDB file not found"])
        stats = get_error_summary(LOG_FILE, verbose=False)
    if os.path.exists(f"{LOG_FILE}_tmp"):
        with open(LOG_FILE, "r") as f:
            lines = [
                x for x in f.readlines() if not x.startswith("<<< PDB file not found")
            ]
        os.remove(LOG_FILE)
        with open(f"{LOG_FILE}_tmp", "a") as f:
            for line in lines:
                f.write(line)
        os.rename(f"{LOG_FILE}_tmp", LOG_FILE)

    if remove_redundancies:
        removed = _remove_database_redundancies(
            OUTPUT_FOLDER, seq_identity_threshold=seq_identity_threshold
        )
        _log_removed(removed, LOG_FILE)

    return get_error_summary(LOG_FILE)


class _PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def pad_collate(self, batch):
        # find longest sequence
        out = {}
        max_len = max(map(lambda x: x["S"].shape[0], batch))

        # pad according to max_len
        to_pad = [max_len - b["S"].shape[0] for b in batch]
        for key in batch[0].keys():
            if key in ["chain_id", "chain_dict"]:
                continue
            out[key] = torch.stack(
                [
                    torch.cat([b[key], torch.zeros((pad, *b[key].shape[1:]))], 0)
                    for b, pad in zip(batch, to_pad)
                ],
                0,
            )
        out["chain_id"] = torch.tensor([b["chain_id"] for b in batch])
        return out

    def __call__(self, batch):
        return self.pad_collate(batch)


def download_data(tag, local_datasets_folder="./data", skip_splitting=False):
    """
    Download a pre-computed dataset with train/test/validation splits

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store bestprot datasets, logs and temporary files
    skip_splitting : bool, default False
        if `True`, skip the split dictionary creation and the file moving steps
    """

    data_path = _download_dataset(tag, local_datasets_folder)
    if not skip_splitting:
        print("We're almost there, just a tiny effort left :-)")
        _split_data(data_path)
        print("-------------------------------------")
    print(
        "Thanks for downloading BestProt, the most complete, user-friendly and loving protein dataset you will ever find! ;-)"
    )

def generate_data(
    tag,
    local_datasets_folder="./data",
    min_length=30,
    max_length=10000,
    resolution_thr=3.5,
    missing_ends_thr=0.3,
    missing_middle_thr=0.1,
    not_filter_methods=False,
    not_remove_redundancies=False,
    skip_splitting=False,
    seq_identity_threshold=0.9,
    n=None,
    force=False,
    split_tolerance=0.2,
    test_split=0.05,
    valid_split=0.05,
    pdb_snapshot=None,
):
    """
    Download and parse PDB files that meet filtering criteria

    The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are
    the following:

    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order, check `sidechain_order()`),
    - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
        zeros to missing values,
    - `'seq'`: a string of length `L` with residue types.

    All errors including reasons for filtering a file out are logged in the log file.

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store bestprot datasets, logs and temporary files
    min_length : int, default 30
        The minimum number of non-missing residues per chain
    max_length : int, default 10000
        The maximum number of residues per chain (set None for no threshold)
    resolution_thr : float, default 3.5
        The maximum resolution
    missing_ends_thr : float, default 0.3
        The maximum fraction of missing residues at the ends
    missing_middle_thr : float, default 0.1
        The maximum fraction of missing values in the middle (after missing ends are disregarded)
    not_filter_methods : bool, default False
        If `False`, only files obtained with X-ray or EM will be processed
    not_remove_redundancies : bool, default False
        If 'False', removes biounits that are doubles of others sequence wise
    skip_splitting : bool, default False
        if `True`, skip the split dictionary creation and the file moving steps
    seq_identity_threshold : float, default 0.9
        The threshold upon which sequences are considered as one and the same (default: 90%)
    n : int, default None
        The number of files to process (for debugging purposes)
    force : bool, default False
        When `True`, rewrite the files if they already exist
    split_tolerance : float, default 0.2
        The tolerance on the split ratio (default 20%)
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    _check_mmseqs()
    filter_methods = not not_filter_methods
    remove_redundancies = not not_remove_redundancies
    tmp_folder = os.path.join(local_datasets_folder, "tmp")
    output_folder = os.path.join(local_datasets_folder, f"bestprot_{tag}")
    log_folder = os.path.join(local_datasets_folder, "logs")
    out_split_dict_folder = os.path.join(output_folder, "splits_dict")

    log_dict = _run_processing(
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        log_folder=log_folder,
        min_length=min_length,
        max_length=max_length,
        resolution_thr=resolution_thr,
        missing_ends_thr=missing_ends_thr,
        missing_middle_thr=missing_middle_thr,
        filter_methods=filter_methods,
        remove_redundancies=remove_redundancies,
        seq_identity_threshold=seq_identity_threshold,
        n=n,
        force=force,
        tag=tag,
        pdb_snapshot=pdb_snapshot,
    )
    if not skip_splitting:
        _get_split_dictionaries(
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            out_split_dict_folder=out_split_dict_folder,
        )

        _split_data(output_folder)
    return log_dict

def split_data(
    tag,
    local_datasets_folder="./data",
    split_tolerance=0.2,
    test_split=0.05,
    valid_split=0.05,
    ignore_existing=False,
):
    """
    Split `bestprot` entry files into training, test and validation.

    ...

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store bestprot datasets, logs and temporary files
    split_tolerance : float, default 0.2
        The tolerance on the split ratio (default 20%)
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)
    ignore_existing : bool, default False
        If `True`, overwrite existing dictionaries for this tag; otherwise, load the existing dictionary

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors
    """

    _check_mmseqs()
    tmp_folder = os.path.join(local_datasets_folder, "tmp")
    output_folder = os.path.join(local_datasets_folder, f"bestprot_{tag}")
    out_split_dict_folder = os.path.join(output_folder, "splits_dict")
    exists = False

    if os.path.exists(out_split_dict_folder):
        if not ignore_existing:
            warnings.warn(f"Found an existing dictionary for tag {tag}. BestProt will load it and ignore the parameters! Run with --ignore_existing to overwrite.")
            exists = True
    if not exists:
        _get_split_dictionaries(
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            out_split_dict_folder=out_split_dict_folder,
        )

    _split_data(output_folder)


class ProteinDataset(Dataset):
    """
    Dataset to load BestProt data

    Saves the model input tensors as pickle files in `features_folder`. When `clustering_dict_path` is provided,
    at each iteration a random bionit from a cluster is sampled.

    Returns dictionaries with the following keys and values (all values are `torch` tensors):

    - `'X'`: 3D coordinates of N, C, Ca, O, `(total_L, 4, 3)`,
    - `'S'`: sequence indices (shape `(total_L)`),
    - `'mask'`: residue mask (0 where coordinates are missing, 1 otherwise), `(total_L)`,
    - `'residue_idx'`: residue indices (from 0 to length of sequence, +100 where chains change), `(total_L)`,
    - `'chain_encoding_all'`: chain indices, `(total_L)`,
    - `'chain_id`': a sampled chain index,
    - `'chain_dict'`: a dictionary of chain ids (keys are chain ids, e.g. `'A'`, values are the indices used in `'chain_id'` and `'chain_encoding_all'` objects)

    You can also choose to include additional features (set in the `node_features_type` parameter):

    - `'sidechain_orientation'`: a unit vector in the direction of the sidechain, `(total_L, 3)`,
    - `'dihedral'`: the dihedral angles, `(total_L, 2)`,
    - `'chemical'`: hydropathy, volume, charge, polarity, acceptor/donor features, `(total_L, 6)`.
    """

    def __init__(
        self,
        dataset_folder,
        features_folder="./data/tmp/",
        clustering_dict_path=None,
        max_length=None,
        rewrite=False,
        use_fraction=1,
        load_to_ram=False,
        debug=False,
        interpolate="none",
        node_features_type="zeros",
        debug_file_path=None,
    ):
        """
        Parameters
        ----------
        dataset_folder : str
            the path to the folder with BestProt format input files (assumes that files are named {biounit_id}.pickle)
        features_folder : str, default "./data/tmp/"
            the path to the folder where the ProteinMPNN features will be saved
        clustering_dict_path : str, optional
            path to the pickled clustering dictionary (keys are cluster ids, values are (biounit id, chain id) tuples)
        max_length : int, optional
            entries with total length of chains larger than `max_length` will be disregarded
        rewrite : bool, default False
            if `False`, existing feature files are not overwritten
        use_fraction : float, default 1
            the fraction of the clusters to use (first N in alphabetic order)
        load_to_ram : bool, default False
            if `True`, the data will be stored in RAM (use with caution! if RAM isn't big enough the machine might crash)
        debug : bool, default False
            only process 1000 files
        interpolate : {"none", "only_middle", "all"}
            `"none"` for no interpolation, `"only_middle"` for only linear interpolation in the middle, `"all"` for linear interpolation + ends generation
        node_features_type : {"zeros", "dihedral", "sidechain", "chemical", or combinations with "+"}
            the type of node features, e.g. "dihedral" or "sidechain+chemical"
        debug_file_path : str, optional
            if not `None`, open this single file instead of loading the dataset
        """

        alphabet = ALPHABET
        self.alphabet_dict = defaultdict(lambda: 0)
        for i, letter in enumerate(alphabet):
            self.alphabet_dict[letter] = i
        self.alphabet_dict["X"] = 0
        self.files = {}  # file path by biounit id
        self.loaded = None
        self.dataset_folder = dataset_folder
        self.features_folder = features_folder
        self.feature_types = node_features_type.split("+")

        if debug_file_path is not None:
            self.dataset_folder = os.path.dirname(debug_file_path)
            debug_file_path = os.path.basename(debug_file_path)

        self.main_atom_dict = defaultdict(lambda: None)
        d1to3 = {v: k for k, v in D3TO1.items()}
        for i, letter in enumerate(alphabet):
            if i == 0:
                continue
            self.main_atom_dict[i] = MAIN_ATOMS[d1to3[letter]]

        # create feature folder if it does not exist
        if not os.path.exists(self.features_folder):
            os.makedirs(self.features_folder)

        self.interpolate = interpolate
        # generate the feature files
        print("Processing files...")
        if debug_file_path is None:
            to_process = os.listdir(dataset_folder)
        else:
            to_process = [debug_file_path]
        if debug:
            to_process = to_process[:1000]
        # output_tuples = [self._process(x, rewrite=rewrite) for x in tqdm(to_process)]
        output_tuples = p_map(lambda x: self._process(x, rewrite=rewrite), to_process)
        # save the file names
        for id, filename in output_tuples:
            self.files[id] = filename
        # filter by length
        if max_length is not None:
            to_remove = []
            for id, file in self.files.items():
                with open(file, "rb") as f:
                    data = pickle.load(f)
                    if len(data["S"]) > max_length:
                        to_remove.append(id)
            for id in to_remove:
                self.files.pop(id)
        # load the clusters
        if clustering_dict_path is not None:
            with open(clustering_dict_path, "rb") as f:
                self.clusters = pickle.load(f)  # list of biounit ids by cluster id
            for key in list(self.clusters.keys()):
                self.clusters[key] = [
                    [x[0].split(".")[0], x[1]]
                    for x in self.clusters[key]
                    if x[0].split(".")[0] in self.files
                ]
                if len(self.clusters[key]) == 0:
                    self.clusters.pop(key)
            self.data = list(self.clusters.keys())
        else:
            self.clusters = None
            self.data = list(self.files.keys())
        # create a smaller datset if necessary
        if use_fraction < 1:
            self.data = sorted(self.data)[: int(len(self.data) * use_fraction)]
        if load_to_ram:
            print("Loading to RAM...")
            self.loaded = {}
            if self.clusters is None:
                for id in tqdm(self.data):
                    with open(self.files[id], "rb") as f:
                        self.loaded[id] = pickle.load(f)
            else:
                pbar = tqdm(total=sum([len(self.clusters[x]) for x in self.data]))
                for cluster in self.data:
                    for id_tuple in self.clusters[cluster]:
                        with open(self.files[id_tuple[0]], "rb") as f:
                            self.loaded[id_tuple[0]] = pickle.load(f)
                            pbar.update(1)
                pbar.close()

    def _interpolate(self, crd_i, mask_i):
        """
        Fill in missing values in the middle with linear interpolation and (if fill_ends is true) build an initialization for the ends

        For the ends, the first 10 residues are 3.6 A apart from each other on a straight line from the last known value away from the center.
        Next they are 3.6 A apart in a random direction.
        """

        if self.interpolate in ["all", "only_middle"]:
            crd_i[(1 - mask_i).astype(bool)] = np.nan
            df = pd.DataFrame(crd_i.reshape((crd_i.shape[0], -1)))
            crd_i = df.interpolate(limit_area="inside").values.reshape(crd_i.shape)
        if self.interpolate == "all":
            non_nans = np.where(~np.isnan(crd_i[:, 0, 0]))[0]
            known_start = non_nans[0]
            known_end = non_nans[-1] + 1
            if known_end < len(crd_i) or known_start > 0:
                center = crd_i[non_nans, 2, :].mean(0)
                if known_start > 0:
                    direction = crd_i[known_start, 2, :] - center
                    direction = direction / linalg.norm(direction)
                    for i in range(0, min(known_start, 10)):
                        crd_i[known_start - i - 1] = (
                            crd_i[known_start - i] + direction * 3.6
                        )
                    for i in range(min(known_start, 10), known_start):
                        v = np.random.rand(3)
                        v = v / linalg.norm(v)
                        crd_i[known_start - i - 1] = crd_i[known_start - i] + v * 3.6
                if known_end < len(crd_i):
                    to_add = len(crd_i) - known_end
                    direction = crd_i[known_end - 1, 2, :] - center
                    direction = direction / linalg.norm(direction)
                    for i in range(0, min(to_add, 10)):
                        crd_i[known_end + i] = (
                            crd_i[known_end + i - 1] + direction * 3.6
                        )
                    for i in range(min(to_add, 10), to_add):
                        v = np.random.rand(3)
                        v = v / linalg.norm(v)
                        crd_i[known_end + i] = crd_i[known_end + i - 1] + v * 3.6
            mask_i = np.ones(mask_i.shape)
        if self.interpolate in ["only_middle"]:
            nan_mask = np.isnan(crd_i)  # in the middle the nans have been interpolated
            mask_i[~np.isnan(crd_i[:, 0, 0])] = 1
            crd_i[nan_mask] = 0
        if self.interpolate == "zeros":
            non_nans = np.where(mask_i != 0)[0]
            known_start = non_nans[0]
            known_end = non_nans[-1] + 1
            mask_i[known_start:known_end] = 1
        return crd_i, mask_i

    def _dihedral_angle(self, crd, msk):
        """Praxeolitic formula
        1 sqrt, 1 cross product"""

        p0 = crd[..., 0, :]
        p1 = crd[..., 1, :]
        p2 = crd[..., 2, :]
        p3 = crd[..., 3, :]

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= np.expand_dims(np.linalg.norm(b1, axis=-1), -1) + 1e-7

        v = b0 - np.expand_dims(np.einsum("bi,bi->b", b0, b1), -1) * b1
        w = b2 - np.expand_dims(np.einsum("bi,bi->b", b2, b1), -1) * b1

        x = np.einsum("bi,bi->b", v, w)
        y = np.einsum("bi,bi->b", np.cross(b1, v), w)
        dh = np.degrees(np.arctan2(y, x))
        dh[1 - msk] = 0
        return dh

    def _dihedral(self, crd, msk):
        """
        Dihedral angles
        """

        angles = []
        # N, C, Ca, O
        # psi
        p = crd[:-1, [0, 2, 1], :]
        p = np.concatenate([p, crd[1:, [0], :]], 1)
        p = np.pad(p, ((0, 1), (0, 0), (0, 0)))
        angles.append(self._dihedral_angle(p, msk))
        # phi
        p = crd[:-1, [1], :]
        p = np.concatenate([p, crd[1:, [0, 2, 1]]], 1)
        p = np.pad(p, ((1, 0), (0, 0), (0, 0)))
        angles.append(self._dihedral_angle(p, msk))
        angles = np.stack(angles, -1)
        return angles

    def _sidechain(self, crd_sc, crd_bb, S):
        """
        Sidechain orientation (defined by the 'main atoms' in the `main_atom_dict` dictionary)
        """

        orientation = np.zeros((crd_sc.shape[0], 3))
        for i in range(1, 21):
            if self.main_atom_dict[i] is not None:
                orientation[S == i] = (
                    crd_sc[S == i, self.main_atom_dict[i], :] - crd_bb[S == i, 2, :]
                )
            else:
                S_mask = S == i
                orientation[S_mask] = np.random.rand(*orientation[S_mask].shape)
        orientation /= (np.expand_dims(linalg.norm(orientation, axis=-1), -1) + 1e-7)
        return orientation

    def _chemical(self, seq):
        """
        Chemical features (hydropathy, volume, charge, polarity, acceptor/donor)
        """

        features = np.array([_PMAP(x) for x in seq])
        return features

    def _process(self, filename, rewrite=False):
        """
        Process a BestProt file and save it as ProteinMPNN features
        """

        input_file = os.path.join(self.dataset_folder, filename)
        output_file = os.path.join(self.features_folder, filename)
        if not rewrite and os.path.exists(output_file):
            pass
        else:
            try:
                with open(input_file, "rb") as f:
                    data = pickle.load(f)
            except:
                print(f'{input_file=}')
            chains = sorted(data.keys())
            X = []
            S = []
            mask = []
            mask_original = []
            chain_encoding_all = []
            residue_idx = []
            node_features = defaultdict(lambda: [])
            last_idx = 0
            chain_dict = {}

            for chain_i, chain in enumerate(chains):

                seq = torch.tensor([self.alphabet_dict[x] for x in data[chain]["seq"]])
                S.append(seq)
                crd_i = data[chain]["crd_bb"]
                mask_i = data[chain]["msk"]
                mask_original.append(deepcopy(mask_i))
                if self.interpolate != "none":
                    crd_i, mask_i = self._interpolate(crd_i, mask_i)
                X.append(crd_i)
                mask.append(mask_i)
                residue_idx.append(torch.arange(len(data[chain]["seq"])) + last_idx)
                last_idx = residue_idx[-1][-1] + 100
                chain_encoding_all.append(torch.ones(len(data[chain]["seq"])) * chain_i)
                chain_dict[chain] = chain_i
                if "dihedral" in self.feature_types:
                    node_features["dihedral"].append(self._dihedral(crd_i, mask_i))
                if "sidechain_orientation" in self.feature_types:
                    node_features["sidechain_orientation"].append(self._sidechain(data[chain]["crd_sc"], crd_i, seq))
                if "chemical" in self.feature_types:
                    node_features["chemical"].append(self._chemical(data[chain]["seq"]))

            out = {}
            out["X"] = torch.from_numpy(np.concatenate(X, 0))
            out["S"] = torch.cat(S)
            out["mask"] = torch.from_numpy(np.concatenate(mask))
            out["mask_original"] = torch.from_numpy(np.concatenate(mask_original))
            out["chain_encoding_all"] = torch.cat(chain_encoding_all)
            out["residue_idx"] = torch.cat(residue_idx)
            out["chain_dict"] = chain_dict
            for key, value_list in node_features.items():
                out[key] = torch.from_numpy(np.concatenate(value_list))
            with open(output_file, "wb") as f:
                pickle.dump(out, f)
        return (
            os.path.basename(filename).split(".")[0],
            output_file,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chain_id = None
        if self.clusters is None:
            id = self.data[idx]  # data is already filtered by length
            chain_id = None
        else:
            cluster = self.data[idx]
            id = None
            while id not in self.files:  # some IDs can be filtered out by length
                chain_n = random.randint(0, len(self.clusters[cluster]) - 1)
                id, chain_id = self.clusters[cluster][
                    chain_n
                ]  # get id and chain from cluster
        if self.loaded is None:
            file = self.files[id]
            with open(file, "rb") as f:
                data = pickle.load(f)
        else:
            data = deepcopy(self.loaded[id])
        if chain_id is None:
            chain_id = random.choice(list(data["chain_dict"].values()))
        else:
            chain_id = data["chain_dict"][chain_id]
        data["chain_id"] = chain_id
        data.pop("chain_dict")
        return data


class ProteinLoader(DataLoader):
    """
    A subclass of `torch.data.utils.DataLoader` tuned for the BestProt dataset

    Creates and iterates over an instance of `ProteinDataset`, omitting the `'chain_dict'` keys. 
    See the `ProteinDataset` docs for more information.
    """

    def __init__(
        self,
        dataset_folder,
        features_folder,
        clustering_dict_path=None,
        max_length=None,
        rewrite=False,
        use_fraction=1,
        load_to_ram=False,
        debug=False,
        interpolate="none",
        node_features_type="zeros",
        batch_size=4,
    ) -> None:
        """
        Parameters
        ----------
        dataset_folder : str
            the path to the folder with BestProt format input files (assumes that files are named {biounit_id}.pickle)
        features_folder : str
            the path to the folder where the ProteinMPNN features will be saved
        clustering_dict_path : str, optional
            path to the pickled clustering dictionary (keys are cluster ids, values are (biounit id, chain id) tuples)
        max_length : int, optional
            entries with total length of chains larger than `max_length` will be disregarded
        rewrite : bool, default False
            if `False`, existing feature files are not overwritten
        use_fraction : float, default 1
            the fraction of the clusters to use (first N in alphabetic order)
        load_to_ram : bool, default False
            if `True`, the data will be stored in RAM (use with caution! if RAM isn't big enough the machine might crash)
        debug : bool, default False
            only process 1000 files
        interpolate : {"none", "only_middle", "all"}
            `"none"` for no interpolation, `"only_middle"` for only linear interpolation in the middle, `"all"` for linear interpolation + ends generation
        node_features_type : {"zeros", "dihedral", "sidechain", "chemical", or combinations with "+"}
            the type of node features, e.g. `"dihedral"` or `"sidechain+chemical"`
        batch_size : int, default 4
            the batch size
        """

        dataset = ProteinDataset(
            dataset_folder=dataset_folder,
            features_folder=features_folder,
            clustering_dict_path=clustering_dict_path,
            max_length=max_length,
            rewrite=rewrite,
            use_fraction=use_fraction,
            load_to_ram=load_to_ram,
            debug=debug,
            interpolate=interpolate,
            node_features_type=node_features_type,
        )
        super().__init__(dataset, collate_fn=_PadCollate(), batch_size=batch_size)


def sidechain_order():
    """
    Get a dictionary of sidechain atom orders

    Returns
    -------
    order_dict : dict
        a dictionary where keys are 3-letter aminoacid codes and values are lists of atom names (in PDB format) that correspond to
        coordinates in the `'crd_sc'` array generated by the `run_processing` function
    """

    return SIDECHAIN_ORDER


def get_error_summary(log_file, verbose=True):
    """
    Get a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    Parameters
    ----------
    log_file : str
        the log file path
    verbose : bool, default True
        if `True`, the statistics are written in the standard output

    Returns
    -------
    log_dict : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors
    """

    stats = defaultdict(lambda: [])
    with open(log_file, "r") as f:
        for line in f.readlines():
            if line.startswith("<<<"):
                stats[line.split(":")[0]].append(line.split(":")[-1].strip())
    keys = sorted(stats.keys(), key=lambda x: stats[x], reverse=True)
    if verbose:
        for key in keys:
            print(f"{key}: {len(stats[key])}")
    return stats

def check_download_tags():
    """
    Get a list of tags available for downloading

    Returns
    -------
    tags : list
        a list of tag names
    """

    folders = _s3list(
        boto3.resource("s3").Bucket("ml4-main-storage"),
        "",
        recursive=False,
        list_objs=False,
    )
    tags_dict = defaultdict(lambda: [])
    for folder in folders:
        folder = folder.key
        if not folder.startswith("bestprot_"):
            print('continue')
            continue
        tag = folder[len("bestprot_"):]
        if tag.endswith("_splits_dict/"):
            tag = tag[: - len("_splits_dict/")]
        else:
            tag = tag.strip("/")
        tags_dict[tag].append(folder)
    return [x for x, v in tags_dict.items() if len(v) == 2]
