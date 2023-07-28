"""Functions for processing PDB files and generating new datasets."""
import multiprocessing
import os
import pickle
import subprocess
from collections import Counter
from datetime import datetime

import editdistance
import numpy as np
import requests
from joblib import Parallel, delayed
from p_tqdm import p_map
from tqdm import tqdm

from proteinflow.data import PDBEntry, SAbDabEntry
from proteinflow.data.utils import PDBError
from proteinflow.download import _load_files
from proteinflow.ligand import _compare_smiles
from proteinflow.logging import _log_exception, _log_removed, get_error_summary


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
    redundancy_thr=0.9,
    n=None,
    force=False,
    tag=None,
    pdb_snapshot=None,
    load_live=False,
    sabdab=False,
    sabdab_data_path=None,
    require_antigen=False,
    max_chains=5,
    pdb_id_list_path=None,
    load_ligands=False,
    require_ligand=False,
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
    redundancy_thr : float, default 0.9
        The threshold upon which sequences are considered as one and the same (default: 90%)
    n : int, default None
        The number of files to process (for debugging purposes)
    force : bool, default False
        When `True`, rewrite the files if they already exist
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
    max_chains : int, default 5
        the maximum number of chains per biounit
    pdb_id_list_path : str, default None
        if provided, get pdb_ids from list (format pdb_id-num example: 1XYZ-1)
    load_ligands: boool, default False
        Whether or not to load the ligands in the pdbs
    require_ligand: bool, default False
        if `True`, only keep files with ligands

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
        f.write(f"    max_chains: {max_chains} \n")
        if remove_redundancies:
            f.write(f"    redundancy_threshold: {redundancy_thr} \n")
        if sabdab:
            f.write(f"    require_antigen: {require_antigen} \n")
            f.write(f"    sabdab_data_path: {sabdab_data_path} \n")
        else:
            f.write(f"    load_live: {load_live} \n")
        f.write("\n")

    def process_f(
        local_paths,
        show_error=False,
        force=True,
        sabdab=False,
        ligand=False,
    ):
        pdb_path, fasta_path = local_paths
        chain_id = None
        if sabdab:
            pdb_path, chain_id = pdb_path
            heavy, light, antigen = chain_id.split("_")
            if heavy == "nan":
                heavy = None
            if light == "nan":
                light = None
            if antigen == "nan":
                antigen = []
            else:
                antigen = antigen.split(" | ")
        fn = os.path.basename(pdb_path)
        pdb_id = fn.split(".")[0]
        if os.path.getsize(pdb_path) > 1e7:
            _log_exception(
                PDBError("PDB / mmCIF file is too large"),
                LOG_FILE,
                pdb_id,
                TMP_FOLDER,
                chain_id=chain_id,
            )
        try:
            # local_path = download_f(pdb_id, s3_client=s3_client, load_live=load_live)
            name = pdb_id if not sabdab else pdb_id + "-" + chain_id
            target_file = os.path.join(OUTPUT_FOLDER, name + ".pickle")
            if not force and os.path.exists(target_file):
                raise PDBError("File already exists")
            if sabdab:
                pdb_entry = SAbDabEntry(
                    pdb_path=pdb_path,
                    heavy_chain=heavy,
                    light_chain=light,
                    antigen_chains=antigen,
                    fasta_path=fasta_path,
                )
            else:
                pdb_entry = PDBEntry(
                    pdb_path=pdb_path, fasta_path=fasta_path, load_ligand=ligand
                )
            # filter and convert
            protein_dict = filter_and_convert(
                pdb_entry,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                max_missing_ends=MISSING_ENDS_THR,
                max_missing_middle=MISSING_MIDDLE_THR,
                load_ligands=ligand,
                require_ligand=require_ligand,
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
            max_chains=max_chains,
            pdb_id_list_path=pdb_id_list_path,
        )
        for id in error_ids:
            with open(LOG_FILE, "a") as f:
                f.write(f"<<< Could not download PDB/mmCIF file: {id} \n")
        # paths = [("data/2c2m-1.pdb.gz", "data/2c2m.fasta")]
        print("Filter and process...")
        _ = p_map(
            lambda x: process_f(x, force=force, sabdab=sabdab, ligand=load_ligands),
            paths,
        )
        # _ = [
        #     process_f(x, force=force, sabdab=sabdab, show_error=True)
        #     for x in tqdm(paths)
        # ]
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
            OUTPUT_FOLDER,
            seq_identity_threshold=redundancy_thr,
            ligand_identity=load_ligands,
        )
        _log_removed(removed, LOG_FILE)

    return get_error_summary(LOG_FILE)


def filter_and_convert(
    pdb_entry,
    min_length=50,
    max_length=150,
    max_missing_ends=5,
    max_missing_middle=5,
    load_ligands: bool = False,
    require_ligand: bool = False,
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
    load_ligands: boool, default False
        Whether or not to load the ligands in the pdbs
    require_ligand: bool, default False
        Whether or not to require the presence of ligands

    Returns
    -------
    ProteinEntry
        The converted ProteinEntry

    """
    pdb_dict = {}
    fasta_dict = pdb_entry.get_fasta()
    loaded_ligands = False
    if load_ligands and pdb_entry.get_ligands() is not None:
        ligand_dict = pdb_entry.get_ligands()
        if len(ligand_dict) > 0:
            loaded_ligands = True
    if require_ligand and not loaded_ligands:
        raise PDBError("No ligands found")

    if len(pdb_entry.get_chains()) == 0:
        raise PDBError("No chains found")

    if pdb_entry.has_unnatural_amino_acids():
        raise PDBError("Unnatural amino acids found")

    for chain in pdb_entry.get_chains():
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
        if loaded_ligands:
            if chain in ligand_dict.keys():
                pdb_dict[chain]["ligand"] = ligand_dict[chain]
        if (pdb_dict[chain]["msk"][start:end] == 0).sum() > max_missing_middle * (
            end - start
        ):
            raise PDBError("Too many missing values in the middle")
    return pdb_dict


def _remove_database_redundancies(
    dir, seq_identity_threshold=0.9, ligand_identity=False
):
    """Remove all biounits in the database that are copies to another biounits in terms of sequence.

    Sequence identity is defined by the 'seq_identity_threshold' parameter for robust detection of sequence similarity (missing residues, point mutations, ...).

    Parameters
    ----------
    dir : str
        the path to the database where all the biounits are stored in pickle files after their processing
    seq_identity_threshold : float, default .9
        the threshold that determines up to what percentage identity sequences are considered as the same

    Returns
    -------
    total_removed : int
        the total number of removed biounits

    """
    all_files = np.array(os.listdir(dir))
    all_pdbs = np.array([file[:4] for file in all_files])
    pdb_counts = Counter(all_pdbs)
    pdbs_to_check = [pdb for pdb in pdb_counts.keys() if pdb_counts[pdb] > 1]
    total_removed = []

    for pdb in tqdm(pdbs_to_check):
        biounits_list = np.array(
            [os.path.join(dir, file) for file in all_files[all_pdbs == pdb]]
        )
        biounits_list = sorted(biounits_list)
        redundancies = _check_biounits(
            biounits_list, seq_identity_threshold, ligand_identity
        )
        if redundancies != []:
            for k in redundancies:
                total_removed.append(os.path.basename(biounits_list[k]).split(".")[0])
                subprocess.run(["rm", biounits_list[k]])

    return total_removed


def _open_pdb(file):
    """Open a PDB file in the pickle format that follows the dwnloading and processing of the database."""
    with open(file, "rb") as f:
        return pickle.load(f)


def _check_biounits(biounits_list, threshold, ligand_identity):
    """Return the indexes of the redundant biounits within the list of files given by `biounits_list`."""
    biounits = [_open_pdb(b) for b in biounits_list]
    indexes = []

    for k, b1 in enumerate(biounits):
        if k not in indexes:
            b1_seqs = [b1[chain]["seq"] for chain in b1.keys()]
            for i, b2 in enumerate(biounits[k + 1 :]):
                if len(b1.keys()) != len(b2.keys()):
                    continue

                b2_seqs = [b2[chain]["seq"] for chain in b2.keys()]
                if ligand_identity:
                    ligs1 = []
                    for chain in b1.keys():
                        if "ligand" in b1[chain].keys():
                            ligs1.append(
                                ".".join(
                                    list([c["smiles"] for c in b1[chain]["ligand"]])
                                )
                            )
                    ligs2 = []
                    for chain in b2.keys():
                        if "ligand" in b2[chain].keys():
                            ligs2.append(
                                ".".join(
                                    list([c["smiles"] for c in b2[chain]["ligand"]])
                                )
                            )
                    equal_ligands = _compare_smiles(ligs1, ligs2, threshold)
                else:
                    equal_ligands = True
                if _compare_seqs(b1_seqs, b2_seqs, threshold) and equal_ligands:
                    indexes.append(k + i + 1)

    return indexes


def _compare_identity(seq, seqs, threshold):
    """Assess whether a sequence is in a list of sequences (in the sense that it shares at least 90% to one of the sequences in the list)."""
    for s in seqs:
        if editdistance.eval(s, seq) / max(len(s), len(seq)) <= (1 - threshold):
            return True

    return False


def _compare_seqs(seqs1, seqs2, threshold):
    """Assess whether 2 lists of sequences contain exactly the same set of sequences."""
    for seq in seqs1:
        if not _compare_identity(seq, seqs2, threshold):
            return False

    for seq in seqs2:
        if not _compare_identity(seq, seqs1, threshold):
            return False

    return True


def _raise_rcsbsearch(e):
    """Raise a RuntimeError if the error is due to rcsbsearch."""
    if "404 Client Error" in str(e):
        raise RuntimeError(
            'Querying rcsbsearch is failing. Please install a version of rcsbsearch where this error is solved:\npython -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"'
        )
    else:
        raise e
