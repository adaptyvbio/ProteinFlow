__pdoc__ = {"utils": False, "scripts": False}
__docformat__ = "numpy"

from proteinflow.constants import _PMAP, ALLOWED_AG_TYPES, ALPHABET, CDR, D3TO1, MAIN_ATOMS
from proteinflow.protein_dataset import ProteinDataset
from proteinflow.protein_loader import ProteinLoader
from proteinflow.utils.filter_database import _remove_database_redundancies
from proteinflow.utils.process_pdb import (
    _align_structure,
    _open_structure,
    PDBError,
    _s3list,
    SIDECHAIN_ORDER,
    _retrieve_fasta_chains,
)
from proteinflow.utils.cluster_and_partition import (
    _build_dataset_partition,
    _check_mmseqs,
)
from proteinflow.utils.split_dataset import _download_dataset, _split_data
from proteinflow.utils.biotite_sse import _annotate_sse
from proteinflow.utils.async_download import _download_s3_parallel

from aiobotocore.session import get_session
import traceback
import shutil
import warnings
import os
import pickle
from collections import defaultdict
from rcsbsearch import Attr
from datetime import datetime
import subprocess
import urllib
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
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from editdistance import eval as edit_distance
import requests
import zipfile
from bs4 import BeautifulSoup
import urllib.request
import string
from einops import rearrange
import tempfile

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


def _log_exception(exception, log_file, pdb_id, tmp_folder, chain_id=None):
    """
    Record the error in the log file
    """

    if chain_id is None:
        _clean(pdb_id, tmp_folder)
    else:
        pdb_id = pdb_id + "-" + chain_id
    if isinstance(exception, PDBError):
        with open(log_file, "a") as f:
            f.write(f"<<< {str(exception)}: {pdb_id} \n")
    else:
        with open(log_file, "a") as f:
            f.write(f"<<< Unknown: {pdb_id} \n")
            f.write(traceback.format_exc())
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
    min_seq_id=0.3,
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
    min_seq_id : float in [0, 1], default 0.3
        minimum sequence identity for `mmseqs`
    """

    sample_file = [x for x in os.listdir(output_folder) if x.endswith(".pickle")][0]
    ind = sample_file.split(".")[0].split("-")[1]
    sabdab = not ind.isnumeric()

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
        min_seq_id=min_seq_id,
        sabdab=sabdab,
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


def _raise_rcsbsearch(e):
    """
    Raise a RuntimeError if the error is due to rcsbsearch
    """

    if "404 Client Error" in str(e):
        raise RuntimeError(
            'Quering rcsbsearch is failing. Please install a version of rcsbsearch where this error is solved:\npython -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"'
        )
    else:
        raise e


def _run_processing(
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
    """
    Download and parse PDB files that meet filtering criteria

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

    LOG_FILE = os.path.join(OUTPUT_FOLDER, f"log.txt")
    print(f"Log file: {LOG_FILE} \n")
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n\n"
    with open(LOG_FILE, "a") as f:
        f.write(date_time)
        if tag is not None:
            f.write(f"tag: {tag} \n\n")

    def process_f(
        local_path,
        show_error=False,
        force=True,
        sabdab=False,
    ):
        chain_id = None
        if sabdab:
            local_path, chain_id = local_path
        fn = os.path.basename(local_path)
        pdb_id = fn.split(".")[0]
        try:
            # local_path = download_f(pdb_id, s3_client=s3_client, load_live=load_live)
            name = pdb_id if not sabdab else pdb_id + "-" + chain_id
            target_file = os.path.join(OUTPUT_FOLDER, name + ".pickle")
            if not force and os.path.exists(target_file):
                raise PDBError("File already exists")
            pdb_dict = _open_structure(
                local_path,
                tmp_folder=TMP_FOLDER,
                sabdab=sabdab,
                chain_id=chain_id,
            )
            # filter and convert
            pdb_dict = _align_structure(
                pdb_dict,
                min_length=MIN_LENGTH,
                max_length=MAX_LENGTH,
                max_missing_ends=MISSING_ENDS_THR,
                max_missing_middle=MISSING_MIDDLE_THR,
                chain_id_string=chain_id,
            )
            # save
            if pdb_dict is not None:
                with open(target_file, "wb") as f:
                    pickle.dump(pdb_dict, f)
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
            tmp_folder=TMP_FOLDER,
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
            with open(LOG_FILE, "r") as f:
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
        with open(LOG_FILE, "r") as f:
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


def _get_pdb_ids(
    resolution_thr=3.5,
    pdb_snapshot=None,
    filter_methods=True,
):
    """
    Get PDB ids from PDB API
    """

    # get filtered PDB ids from PDB API
    pdb_ids = (
        Attr("rcsb_entry_info.selected_polymer_entity_types")
        .__eq__("Protein (only)")
        .or_("rcsb_entry_info.polymer_composition")
        .__eq__("protein/oligosaccharide")
    )
    # if include_na:
    #     pdb_ids = pdb_ids.or_('rcsb_entry_info.polymer_composition').in_(["protein/NA", "protein/NA/oligosaccharide"])

    if resolution_thr is not None:
        pdb_ids = pdb_ids.and_("rcsb_entry_info.resolution_combined").__le__(
            resolution_thr
        )
    if filter_methods:
        pdb_ids = pdb_ids.and_("exptl.method").in_(
            ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
        )
    pdb_ids = pdb_ids.exec("assembly")

    ordered_folders = [
        x.key.strip("/")
        for x in _s3list(
            boto3.resource("s3", config=Config(signature_version=UNSIGNED)).Bucket(
                "pdbsnapshots"
            ),
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
    return ordered_folders, pdb_ids


def _download_live(id, tmp_folder):
    """
    Download a PDB file from the PDB website
    """

    pdb_id, biounit = id.split("-")
    filenames = {
        "cif": f"{pdb_id}-assembly{biounit}.cif.gz",
        "pdb": f"{pdb_id}.pdb{biounit}.gz",
    }
    for t in filenames:
        local_path = os.path.join(tmp_folder, f"{pdb_id}-{biounit}") + f".{t}.gz"
        try:
            url = f"https://files.rcsb.org/download/{filenames[t]}"
            response = requests.get(url)
            open(local_path, "wb").write(response.content)
            return local_path
        except:
            pass
    return id


def _download_fasta_f(pdb_id, datadir):
    """
    Download a fasta file from the PDB website
    """

    downloadurl = "https://www.rcsb.org/fasta/entry/"
    pdbfn = pdb_id + "/download"
    outfnm = os.path.join(datadir, f"{pdb_id.lower()}.fasta")

    url = downloadurl + pdbfn
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm

    except Exception as err:
        # print(str(err), file=sys.stderr)
        return None


def _load_pdb(
    resolution_thr=3.5,
    pdb_snapshot=None,
    filter_methods=True,
    n=None,
    tmp_folder="data/tmp",
    load_live=False,
):
    """
    Download filtered PDB files and return a list of local file paths
    """

    ordered_folders, pdb_ids = _get_pdb_ids(
        resolution_thr=resolution_thr,
        pdb_snapshot=pdb_snapshot,
        filter_methods=filter_methods,
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        print("Getting a file list...")
        ids = []
        for i, x in enumerate(tqdm(pdb_ids)):
            ids.append(x)
            if n is not None and i == n:
                break
        print("Downloading fasta files...")
        pdbs = set([x.split("-")[0] for x in ids])
        future_to_key = {
            executor.submit(
                lambda x: _download_fasta_f(x, datadir=tmp_folder), key
            ): key
            for key in pdbs
        }
        _ = [
            x.result()
            for x in tqdm(futures.as_completed(future_to_key), total=len(pdbs))
        ]

    # _ = [process_f(x, force=force, load_live=load_live) for x in tqdm(ids)]
    print("Downloading structure files...")
    paths = _download_s3_parallel(
        pdb_ids=ids, tmp_folder=tmp_folder, snapshots=[ordered_folders[0]]
    )
    paths = [item for sublist in paths for item in sublist]
    error_ids = [x for x in paths if not x.endswith(".gz")]
    paths = [x for x in paths if x.endswith(".gz")]
    if load_live:
        print("Downloading newest structure files...")
        live_paths = p_map(
            lambda x: _download_live(x, tmp_folder=tmp_folder), error_ids
        )
        error_ids = []
        for x in live_paths:
            if x.endswith(".gz"):
                paths.append(x)
            else:
                error_ids.append(x)
    return paths, error_ids


def _make_sabdab_html(method, resolution_thr):
    """
    Make a URL for SAbDab search
    """

    html = f"https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?ABtype=All&method={'+'.join(method)}&species=All&resolution={resolution_thr}&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=All&isin_covabdab=All&isin_therasabdab=All&chothiapos=&restype=ALA&field_0=Antigens&keyword_0=#downloads"
    return html


def _load_sabdab(
    resolution_thr=3.5,
    filter_methods=True,
    pdb_snapshot=None,
    tmp_folder="data/tmp",
    sabdab_data_path=None,
    require_antigen=True,
    n=None,
):
    """
    Download filtered SAbDab files and return a list of local file paths
    """

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    if pdb_snapshot is not None:
        pdb_snapshot = datetime.strptime(pdb_snapshot, "%Y%m%d")
    if filter_methods:
        methods = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    else:
        methods = ["All"]
    methods = [x.split() for x in methods]
    if sabdab_data_path is None:
        for method in methods:
            html = _make_sabdab_html(method, resolution_thr)
            page = requests.get(html)
            soup = BeautifulSoup(page.text, "html.parser")
            try:
                zip_ref = soup.find_all(
                    lambda t: t.name == "a" and t.text.startswith("zip")
                )[0]["href"]
                zip_ref = "https://opig.stats.ox.ac.uk" + zip_ref
            except:
                error = soup.find_all(
                    lambda t: t.name == "h1" and t.text.startswith("Internal")
                )
                if len(error) > 0:
                    raise RuntimeError(
                        "Internal SAbDab server error -> try again in some time"
                    )
                raise RuntimeError("No link found")
            print(f'Downloading {" ".join(method)} structure files...')
            subprocess.run(
                [
                    "wget",
                    zip_ref,
                    "-O",
                    os.path.join(tmp_folder, f"pdb_{'_'.join(method)}.zip"),
                ]
            )
        paths = [
            os.path.join(tmp_folder, f"pdb_{'_'.join(method)}.zip")
            for method in methods
        ]
    else:
        paths = [sabdab_data_path]
    ids = []
    pdb_ids = []
    error_ids = []
    print("Moving files...")
    for path in paths:
        if not os.path.isdir(path):
            if not path.endswith(".zip"):
                raise ValueError("SAbDab data path should be a zip file or a directory")
            dir_path = path[:-4]
            print(f"Unzipping {path}...")
            with zipfile.ZipFile(path, "r") as zip_ref:
                for member in tqdm(zip_ref.infolist()):
                    try:
                        zip_ref.extract(member, dir_path)
                    except zipfile.error as e:
                        pass
            if sabdab_data_path is None:
                os.remove(path)
        else:
            dir_path = path
        print("Filtering...")
        summary_path = None
        for file in os.listdir(dir_path):
            if file.endswith(".tsv"):
                summary_path = os.path.join(dir_path, file)
                break
        if summary_path is None:
            raise ValueError("Summary file not found")
        summary = pd.read_csv(summary_path, sep="\t")
        # check antigen type
        summary = summary[summary["antigen_type"].isin(ALLOWED_AG_TYPES)]
        # filter out structures with repeating chains
        summary = summary[summary["antigen_chain"] != summary["Hchain"]]
        summary = summary[summary["antigen_chain"] != summary["Lchain"]]
        summary = summary[summary["Lchain"] != summary["Hchain"]]
        # optional filters
        if require_antigen:
            summary = summary[~summary["antigen_chain"].isna()]
        if pdb_snapshot is not None:
            date = pd.to_datetime(summary["date"], format="%m/%d/%Y")
            summary = summary[date <= pdb_snapshot]
        if sabdab_data_path is not None:
            summary.loc[summary["resolution"] == "NOT", "resolution"] = 0
            if summary["resolution"].dtype != float:
                summary["resolution"] = summary["resolution"].str.split(", ").str[0]
            summary = summary[summary["resolution"].astype(float) <= resolution_thr]
            if filter_methods:
                summary = summary[
                    summary["method"].isin([" ".join(m) for m in methods])
                ]
        if n is not None:
            summary = summary.iloc[:n]
        ids_method = summary["pdb"].unique().tolist()
        for id in tqdm(ids_method):
            pdb_path = os.path.join(dir_path, "chothia", f"{id}.pdb")
            try:
                if sabdab_data_path is None or not os.path.isdir(sabdab_data_path):
                    shutil.move(pdb_path, os.path.join(tmp_folder, f"{id}.pdb"))
                else:
                    shutil.copy(pdb_path, os.path.join(tmp_folder, f"{id}.pdb"))
            except FileNotFoundError:
                error_ids.append(id)
        if sabdab_data_path is None or sabdab_data_path.endswith(".zip"):
            shutil.rmtree(dir_path)
        ids_full = summary.apply(
            lambda x: (x["pdb"], f"{x['Hchain']}_{x['Lchain']}_{x['antigen_chain']}"),
            axis=1,
        ).tolist()
        ids += ids_full
        pdb_ids += ids_method
    print("Downloading fasta files...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # pdb_ids = ["6tkb"]
        future_to_key = {
            executor.submit(
                lambda x: _download_fasta_f(x, datadir=tmp_folder), key
            ): key
            for key in pdb_ids
        }
        _ = [
            x.result()
            for x in tqdm(futures.as_completed(future_to_key), total=len(pdb_ids))
        ]
    paths = [(os.path.join(tmp_folder, f"{x[0]}.pdb"), x[1]) for x in ids]
    return paths, error_ids


def _load_files(
    resolution_thr=3.5,
    pdb_snapshot=None,
    filter_methods=True,
    n=None,
    tmp_folder="data/tmp",
    load_live=False,
    sabdab=False,
    sabdab_data_path=None,
    require_antigen=False,
):
    """
    Download filtered structure files and return a list of local file paths
    """

    if sabdab:
        out = _load_sabdab(
            resolution_thr=resolution_thr,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            tmp_folder=tmp_folder,
            sabdab_data_path=sabdab_data_path,
            require_antigen=require_antigen,
            n=n,
        )
    else:
        out = _load_pdb(
            resolution_thr=resolution_thr,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            tmp_folder=tmp_folder,
            load_live=load_live,
            n=n,
        )
    return out


def download_data(tag, local_datasets_folder="./data", skip_splitting=False):
    """
    Download a pre-computed dataset with train/test/validation splits

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store proteinflow datasets, logs and temporary files
    skip_splitting : bool, default False
        if `True`, skip the split dictionary creation and the file moving steps
    """

    sabdab_data_path = _download_dataset(tag, local_datasets_folder)
    if not skip_splitting:
        _split_data(sabdab_data_path)


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
    load_live=False,
    min_seq_id=0.3,
    sabdab=False,
    sabdab_data_path=None,
    require_antigen=False,
    exclude_chains=None,
    exclude_threshold=0.7,
    exclude_clusters=False,
    exclude_based_on_cdr=None,
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

    When creating a SAbDab dataset, an additional key is added to the dictionary:
    - `'cdr'`: a `'numpy'` array of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
        and non-CDR residues are marked with `'-'`.

    PDB datasets are split into clusters according to sequence identity and SAbDab datasets are split according to CDR similarity.

    All errors including reasons for filtering a file out are logged in the log file.

    For more information on the splitting procedure and options, check out the `proteinflow.split_data` documentation.

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store proteinflow datasets, logs and temporary files
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
    pdb_snapshot : str, optional
        the PDB snapshot to use, by default the latest is used
    load_live : bool, default False
        if `True`, load the files that are not in the latest PDB snapshot from the PDB FTP server (forced to `False` if `pdb_snapshot` is not `None`)
    min_seq_id : float in [0, 1], default 0.3
        minimum sequence identity for `mmseqs`
    sabdab : bool, default False
        if `True`, download the SAbDab database instead of PDB
    sabdab_data_path : str, optional
        path to a zip file or a directory containing SAbDab files (only used if `sabdab` is `True`)
    require_antigen : bool, default False
        if `True`, only use SAbDab files with an antigen
    exclude_chains : list of str, optional
        a list of chains (`{pdb_id}-{chain_id}`) to exclude from the splitting (e.g. `["1A2B-A", "1A2B-B"]`); chain id is the author chain id
    exclude_threshold : float in [0, 1], default 0.7
        the sequence similarity threshold for excluding chains
    exclude_clusters : bool, default False
        if `True`, exclude clusters that contain chains similar to chains in the `exclude_chains` list
    exclude_based_on_cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
        if given and `exclude_clusters` is `True` + the dataset is SAbDab, exclude files based on only the given CDR clusters

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    filter_methods = not not_filter_methods
    remove_redundancies = not not_remove_redundancies
    tmp_id = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(5)
    )
    tmp_folder = os.path.join("", "tmp", tag + tmp_id)
    output_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")
    out_split_dict_folder = os.path.join(output_folder, "splits_dict")

    if force and os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    log_dict = _run_processing(
        tmp_folder=tmp_folder,
        output_folder=output_folder,
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
        load_live=load_live,
        sabdab=sabdab,
        sabdab_data_path=sabdab_data_path,
        require_antigen=require_antigen,
    )
    if not skip_splitting:
        split_data(
            tag=tag,
            local_datasets_folder=local_datasets_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            ignore_existing=True,
            min_seq_id=min_seq_id,
            exclude_chains=exclude_chains,
            exclude_threshold=exclude_threshold,
            exclude_clusters=exclude_clusters,
            exclude_based_on_cdr=exclude_based_on_cdr,
        )
    shutil.rmtree(tmp_folder)
    return log_dict


def _get_excluded_files(
    tag, local_datasets_folder, tmp_folder, exclude_chains, exclude_threshold
):
    """
    Get a list of files to exclude from the dataset.

    Biounits are excluded if they contain chains that are too similar
    (above `exclude_threshold`) to chains in the list of excluded chains.

    Parameters
    ----------
    tag : str
        the name of the dataset
    local_datasets_folder : str
        the path to the folder that stores proteinflow datasets
    tmp_folder : str
        the path to the folder that stores temporary files
    exclude_chains : list of str, optional
        a list of chains (`{pdb_id}-{chain_id}`) to exclude from the splitting (e.g. `["1A2B-A", "1A2B-B"]`); chain id is the author chain id
    exclude_threshold : float in [0, 1], default 0.7
        the sequence similarity threshold for excluding chains
    """

    # download fasta files for excluded chains
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    sequences = []
    for chain in exclude_chains:
        pdb_id, chain_id = chain.split("-")
        downloadurl = "https://www.rcsb.org/fasta/entry/"
        pdbfn = pdb_id + "/download"
        outfnm = os.path.join(tmp_folder, f"{pdb_id.lower()}.fasta")
        url = downloadurl + pdbfn
        urllib.request.urlretrieve(url, outfnm)
        chains = _retrieve_fasta_chains(outfnm)
        sequences.append(chains[chain_id])
        os.remove(outfnm)

    # iterate over files in the dataset to check similarity
    print("Checking excluded chains similarity...")
    exclude_biounits = []
    for fn in tqdm(
        os.listdir(os.path.join(local_datasets_folder, f"proteinflow_{tag}"))
    ):
        if not fn.endswith(".pickle"):
            continue
        fp = os.path.join(local_datasets_folder, f"proteinflow_{tag}", fn)
        with open(fp, "rb") as f:
            entry = pickle.load(f)
        break_flag = False
        for chain, chain_data in entry.items():
            for seq in sequences:
                if (
                    edit_distance(seq, chain_data["seq"]) / len(seq)
                    < 1 - exclude_threshold
                ):
                    exclude_biounits.append(fn)
                    break_flag = True
                    break
            if break_flag:
                break

    # return list of biounits to exclude
    return exclude_biounits


def split_data(
    tag,
    local_datasets_folder="./data",
    split_tolerance=0.2,
    test_split=0.05,
    valid_split=0.05,
    ignore_existing=False,
    min_seq_id=0.3,
    exclude_chains=None,
    exclude_threshold=0.7,
    exclude_clusters=False,
    exclude_based_on_cdr=None,
):
    """
    Split `proteinflow` entry files into training, test and validation.

    Our splitting algorithm has two objectives: achieving minimal data leakage and balancing the proportion of
    single chain, homomer and heteromer entries.

    It follows these steps:

    1. cluster chains by sequence identity,
    2. generate a graph where nodes are the clusters and edges are protein-protein interactions between chains
    from those clusters,
    3. split connected components of the graph into training, test and validation subsets while keeping the proportion
    of single chains, homomers and heteromers close to that in the full dataset (within `split_tolerance`).

    For SAbDab datasets, instead of sequence identity, we use CDR cluster identity to cluster chains. We also connect the clusters
    in the graph if CDRs from those clusters are seen in the same PDB.

    Biounits that contain chains similar to those in the `exclude_chains` list (with `exclude_threshold` sequence identity)
    are excluded from the splitting and placed into a separate folder. If you want to exclude clusters containing those chains
    as well, set `exclude_clusters` to `True`. For SAbDab datasets, you can additionally choose to only exclude based on specific
    CDR clusters. To do so, set `exclude_based_on_cdr` to the CDR type.

    Parameters
    ----------
    tag : str
        the name of the dataset to load
    local_datasets_folder : str, default "./data"
        the path to the folder that will store proteinflow datasets, logs and temporary files
    split_tolerance : float, default 0.2
        The tolerance on the split ratio (default 20%)
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)
    ignore_existing : bool, default False
        If `True`, overwrite existing dictionaries for this tag; otherwise, load the existing dictionary
    min_seq_id : float in [0, 1], default 0.3
        minimum sequence identity for `mmseqs`
    exclude_chains : list of str, optional
        a list of chains (`{pdb_id}-{chain_id}`) to exclude from the splitting (e.g. `["1A2B-A", "1A2B-B"]`); chain id is the author chain id
    exclude_threshold : float in [0, 1], default 0.7
        the sequence similarity threshold for excluding chains
    exclude_clusters : bool, default False
        if `True`, exclude clusters that contain chains similar to chains in the `exclude_chains` list
    exclude_based_on_cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
        if given and `exclude_clusters` is `True` + the dataset is SAbDab, exclude files based on only the given CDR clusters

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors
    """

    if exclude_chains is None or len(exclude_chains) == 0:
        excluded_biounits = []
    else:
        excluded_biounits = _get_excluded_files(
            tag,
            local_datasets_folder,
            os.path.join(local_datasets_folder, "tmp"),
            exclude_chains,
            exclude_threshold,
        )

    tmp_folder = os.path.join(local_datasets_folder, "tmp")
    output_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")
    out_split_dict_folder = os.path.join(output_folder, "splits_dict")
    exists = False

    if os.path.exists(out_split_dict_folder):
        if not ignore_existing:
            warnings.warn(
                f"Found an existing dictionary for tag {tag}. proteinflow will load it and ignore the parameters! Run with --ignore_existing to overwrite."
            )
            exists = True
    if not exists:
        _check_mmseqs()
        _get_split_dictionaries(
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            out_split_dict_folder=out_split_dict_folder,
            min_seq_id=min_seq_id,
        )

    _split_data(
        output_folder, excluded_biounits, exclude_clusters, exclude_based_on_cdr
    )


def unsplit_data(
    tag,
    local_datasets_folder="./data",
):
    """
    Move files from train, test, validation and excluded folders back into the main folder

    Parameters
    ----------
    tag : str
        the name of the dataset
    local_datasets_folder : str, default "./data"
        the path to the folder that stores proteinflow datasets
    """

    for folder in ["excluded", "train", "test", "valid"]:
        if not os.path.exists(
            os.path.join(local_datasets_folder, f"proteinflow_{tag}", folder)
        ):
            continue
        for file in os.listdir(
            os.path.join(local_datasets_folder, f"proteinflow_{tag}", folder)
        ):
            shutil.move(
                os.path.join(local_datasets_folder, f"proteinflow_{tag}", folder, file),
                os.path.join(local_datasets_folder, f"proteinflow_{tag}", file),
            )
        shutil.rmtree(os.path.join(local_datasets_folder, f"proteinflow_{tag}", folder))


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
    Get a dictionary where keys are recognized exception names and values are lists of PDB ids that caused the exceptions

    Parameters
    ----------
    log_file : str
        the log file path
    verbose : bool, default True
        if `True`, the statistics are written in the standard output

    Returns
    -------
    log_dict : dict
        a dictionary where keys are recognized exception names and values are lists of PDB ids that caused the exceptions
    """

    stats = defaultdict(lambda: [])
    with open(log_file, "r") as f:
        for line in f.readlines():
            if line.startswith("<<<"):
                stats[line.split(":")[0]].append(line.split(":")[-1].strip())
    if verbose:
        keys = sorted(stats.keys(), key=lambda x: len(stats[x]), reverse=True)
        for key in keys:
            print(f"{key}: {len(stats[key])}")
        print(f"Total exceptions: {sum([len(x) for x in stats.values()])}")
    return stats


def check_pdb_snapshots():
    """
    Get a list of PDB snapshots available for downloading

    Returns
    -------
    snapshots : list
        a list of snapshot names
    """

    folders = _s3list(
        boto3.resource("s3", config=Config(signature_version=UNSIGNED)).Bucket(
            "pdbsnapshots"
        ),
        "",
        recursive=False,
        list_objs=False,
    )
    return [x.key.strip("/") for x in folders]


def check_download_tags():
    """
    Get a list of tags available for downloading

    Returns
    -------
    tags : list
        a list of tag names
    """

    folders = _s3list(
        boto3.resource("s3", config=Config(signature_version=UNSIGNED)).Bucket(
            "proteinflow-datasets"
        ),
        "",
        recursive=False,
        list_objs=False,
    )
    return [x.key.strip("/") for x in folders]
