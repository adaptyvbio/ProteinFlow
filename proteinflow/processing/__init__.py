import os
import pickle
from datetime import datetime

import numpy as np
from editdistance import eval as edit_distance
from p_tqdm import p_map
from rcsbsearch import Attr
from tqdm import tqdm

from proteinflow.constants import ALLOWED_AG_TYPES, SIDECHAIN_ORDER
from proteinflow.data import PDBEntry, ProteinEntry, SAbDabEntry
from proteinflow.download import _load_files
from proteinflow.logging import _log_exception, _log_removed, get_error_summary
from proteinflow.pdb import _align_structure, _open_structure
from proteinflow.protein_dataset import (
    ProteinDataset,
    _download_dataset,
    _remove_database_redundancies,
    _split_data,
)
from proteinflow.protein_loader import ProteinLoader
from proteinflow.sequences import _retrieve_fasta_chains
from proteinflow.utils.boto_utils import _download_s3_parallel, _s3list
from proteinflow.utils.cluster_and_partition import (
    _build_dataset_partition,
    _check_mmseqs,
)
from proteinflow.utils.common_utils import (
    PDBError,
    _make_sabdab_html,
    _raise_rcsbsearch,
)


def run_processing(
    tmp_folder="./data/tmp_pdb",
    output_folder="./data/pdb",
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
    load_live=False,
    sabdab=False,
    sabdab_data_path=None,
    require_antigen=False,
):
    """Download and parse PDB files that meet filtering criteria.

    The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are
    the following:

    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order, check `sidechain_order()`),
    - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
        zeros to missing values,
    - `'seq'`: a string of length `L` with residue types.

    When creating a SAbDab dataset, an additional key is added to the dictionary:
    - `'cdr'`: a `'numpy'` array of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
        and non-CDR residues are marked with `'-'`.

    All errors including reasons for filtering a file out are logged in a log file.

    Parameters
    ----------
    tmp_folder : str, default "./data/tmp_pdb"
        The folder where temporary files will be saved
    output_folder : str, default "./data/pdb"
        The folder where the output files will be saved
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
        If `True`, removes biounits that are doubles of others sequence wise
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
        The folder where the dictionaries containing the train/validation/test splits information will be saved
    tag : str, optional
        A tag to add to the log file
    pdb_snapshot : str, optional
        the PDB snapshot to use, by default the latest is used (if `sabdab` is `True`, you can use any date in the format YYYYMMDD as a cutoff)
    load_live : bool, default False
        if `True`, load the files that are not in the latest PDB snapshot from the PDB FTP server (forced to `False` if `pdb_snapshot` is not `None`)
    sabdab : bool, default False
        if `True`, download the SAbDab database instead of PDB
    sabdab_data_path : str, optional
        path to a zip file or a directory containing SAbDab files (only used if `sabdab` is `True`)
    require_antigen : bool, default False
        if `True`, only keep files with antigen chains (only used if `sabdab` is `True`)

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    TMP_FOLDER = tmp_folder
    OUTPUT_FOLDER = output_folder
    MIN_LENGTH = min_length
    MAX_LENGTH = max_length
    RESOLUTION_THR = resolution_thr
    MISSING_ENDS_THR = missing_ends_thr
    MISSING_MIDDLE_THR = missing_middle_thr

    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    LOG_FILE = os.path.join(OUTPUT_FOLDER, "log.txt")
    print(f"Log file: {LOG_FILE} \n")
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n\n"
    with open(LOG_FILE, "a") as f:
        f.write(date_time)
        if tag is not None:
            f.write(f"tag: {tag} \n")
        f.write(f"    min_length: {min_length} \n")
        f.write(f"    max_length: {max_length} \n")
        f.write(f"    resolution_thr: {resolution_thr} \n")
        f.write(f"    missing_ends_thr: {missing_ends_thr} \n")
        f.write(f"    missing_middle_thr: {missing_middle_thr} \n")
        f.write(f"    filter_methods: {filter_methods} \n")
        f.write(f"    remove_redundancies: {remove_redundancies} \n")
        f.write(f"    sabdab: {sabdab} \n")
        f.write(f"    pdb_snapshot: {pdb_snapshot} \n")
        if remove_redundancies:
            f.write(f"    seq_identity_threshold: {seq_identity_threshold} \n")
        if sabdab:
            f.write(f"    require_antigen: {require_antigen} \n")
            f.write(f"    sabdab_data_path: {sabdab_data_path} \n")
        else:
            f.write(f"    load_live: {load_live} \n")
        f.write("\n")

    def process_f(
        local_path,
        show_error=False,
        force=True,
        sabdab=False,
    ):
        chain_id = None
        if sabdab:
            local_path, chain_id = local_path
            heavy, light, antigen = ...
        fn = os.path.basename(local_path)
        pdb_id = fn.split(".")[0]
        try:
            # local_path = download_f(pdb_id, s3_client=s3_client, load_live=load_live)
            name = pdb_id if not sabdab else pdb_id + "-" + chain_id
            target_file = os.path.join(OUTPUT_FOLDER, name + ".pickle")
            if not force and os.path.exists(target_file):
                raise PDBError("File already exists")
            if sabdab:
                pdb_entry = SAbDabEntry(
                    pdb_path=local_path,
                    heavy=heavy,
                    light=light,
                    antigen=antigen,
                    fasta_path=...,
                )
            else:
                pdb_entry = PDBEntry(pdb_path=local_path, fasta_path=...)
            # filter and convert
            protein_dict = filter_and_convert(
                pdb_entry,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                max_missing_ends=MISSING_ENDS_THR,
                max_missing_middle=MISSING_MIDDLE_THR,
            )
            # save
            with open(target_file, "wb") as f:
                pickle.dump(protein_dict, f)
        except Exception as e:
            if show_error:
                raise e
            else:
                _log_exception(e, LOG_FILE, pdb_id, TMP_FOLDER, chain_id=chain_id)

    try:
        paths, error_ids = _load_files(
            resolution_thr=RESOLUTION_THR,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            n=n,
            local_folder=TMP_FOLDER,
            load_live=load_live,
            sabdab=sabdab,
            sabdab_data_path=sabdab_data_path,
            require_antigen=require_antigen,
        )
        for id in error_ids:
            with open(LOG_FILE, "a") as f:
                f.write(f"<<< Could not download PDB/mmCIF file: {id} \n")
        # paths = [(os.path.join(TMP_FOLDER, "6tkb.pdb"), "H_L_nan")]
        print("Filter and process...")
        _ = p_map(lambda x: process_f(x, force=force, sabdab=sabdab), paths)
        # _ = [process_f(x, force=force, sabdab=sabdab, show_error=True) for x in tqdm(paths)]
    except Exception as e:
        _raise_rcsbsearch(e)

    stats = get_error_summary(LOG_FILE, verbose=False)
    not_found_error = "<<< PDB / mmCIF file downloaded but not found"
    if not sabdab:
        while not_found_error in stats:
            with open(LOG_FILE) as f:
                lines = [x for x in f.readlines() if not x.startswith(not_found_error)]
            os.remove(LOG_FILE)
            with open(f"{LOG_FILE}_tmp", "a") as f:
                for line in lines:
                    f.write(line)
            if sabdab:
                paths = [
                    (
                        os.path.join(TMP_FOLDER, x.split("-")[0] + ".pdb"),
                        x.split("-")[1],
                    )
                    for x in stats[not_found_error]
                ]
            else:
                paths = stats[not_found_error]
            _ = p_map(lambda x: process_f(x, force=force, sabdab=sabdab), paths)
            stats = get_error_summary(LOG_FILE, verbose=False)
    if os.path.exists(f"{LOG_FILE}_tmp"):
        with open(LOG_FILE) as f:
            lines = [x for x in f.readlines() if not x.startswith(not_found_error)]
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


def filter_and_convert(
    pdb_entry,
    min_length=50,
    max_length=150,
    max_missing_ends=5,
    max_missing_middle=5,
):
    """Filter and convert a PDBEntry to a ProteinEntry.

    Parameters
    ----------
    pdb_entry : PDBEntry
        PDBEntry to be converted
    min_length : int, default 50
        Minimum total length of the protein sequence
    max_length : int, default 150
        Maximum total length of the protein sequence
    missing_ends_thr : float, default 0.3
        The maximum fraction of missing residues at the ends
    missing_middle_thr : float, default 0.1
        The maximum fraction of missing residues in the middle (after missing ends are disregarded)

    Returns
    -------
    ProteinEntry
        The converted ProteinEntry

    """
    pdb_dict = {}
    fasta_dict = pdb_entry.get_fasta()

    if len(pdb_entry.get_chains()) == 0:
        raise PDBError("No chains found")

    if pdb_entry.has_unnatural_amino_acids():
        raise PDBError("Unnatural amino acids found")

    for (chain,) in pdb_entry.get_chains():
        pdb_dict[chain] = {}
        chain_crd = pdb_entry.get_sequence_df(chain)
        fasta_seq = fasta_dict[chain]

        if len(chain_crd) / len(fasta_seq) < 1 - (
            max_missing_ends + max_missing_middle
        ):
            raise PDBError("Too many missing values in total")

        # align fasta and pdb and check criteria)
        mask = pdb_entry.get_mask([chain])[chain]
        known_ind = np.where(mask == 1)[0]
        start, end = known_ind[0], known_ind[-1] + 1
        if start + (len(mask) - end) > max_missing_ends * len(mask):
            raise PDBError("Too many missing values in the ends")
        if (1 - mask)[start:end].sum() > max_missing_middle * (end - start):
            raise PDBError("Too many missing values in the middle")
        if isinstance(pdb_entry, SAbDabEntry):
            pdb_dict[chain]["cdr"] = pdb_entry.get_cdr([chain])[chain]
        pdb_dict[chain]["seq"] = fasta_seq
        pdb_dict[chain]["msk"] = mask
        if min_length is not None and mask.sum() < min_length:
            raise PDBError("Sequence is too short")
        if max_length is not None and len(mask) > max_length:
            raise PDBError("Sequence is too long")

        # go over rows of coordinates
        crd_arr = pdb_entry.get_coordinates_array(chain)

        pdb_dict[chain]["crd_bb"] = crd_arr[:, :4, :]
        pdb_dict[chain]["crd_sc"] = crd_arr[:, 4:, :]
        pdb_dict[chain]["msk"][(pdb_dict[chain]["crd_bb"] == 0).sum(-1).sum(-1) > 0] = 0
        if (pdb_dict[chain]["msk"][start:end] == 0).sum() > max_missing_middle * (
            end - start
        ):
            raise PDBError("Too many missing values in the middle")
    return pdb_dict
