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
import os
import pickle
from collections import defaultdict
import boto3
from p_tqdm import p_map
from rcsbsearch import Attr
from datetime import datetime
import subprocess


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


def download_data(tag, local_datasets_folder="./data"):
    """
    Download a pre-computed dataset with train/test/validation splits

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store bestprot datasets, logs and temporary files
    """

    data_path = _download_dataset(tag, local_datasets_folder)
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
    filter_methods=True,
    remove_redundancies=False,
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
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)
    out_split_dict_folder : str, default "./data/dataset_splits_dict"
        The folder where the dictionaries containing the train/validation/test splits information will be saved"

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    _check_mmseqs()
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
