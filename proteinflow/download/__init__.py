import os
import shutil
import subprocess
import urllib
import urllib.request
import zipfile
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from bs4 import BeautifulSoup
from p_tqdm import p_map
from rcsbsearch import Attr
from tqdm import tqdm

from proteinflow.constants import ALLOWED_AG_TYPES
from proteinflow.utils.boto_utils import _download_s3_parallel, _s3list
from proteinflow.utils.common_utils import _make_sabdab_html


def _download_file(url, local_path):
    """Download a file from a URL to a local path"""
    response = requests.get(url)
    open(local_path, "wb").write(response.content)


def download_pdb(pdb_id, local_folder=".", sabdab=False):
    """
    Download a PDB file from the RCSB PDB database.

    Parameters
    ----------
    pdb_id : str
        PDB ID of the protein to download, can include a biounit index separated
        by a dash (e.g. "1a0a", "1a0a-1")
    local_folder : str, default "."
        Folder to save the downloaded file to
    sabdab : bool, default False
        If True, download from the SAbDab database (Chothia style) instead of RCSB PDB

    Returns
    -------
    local_path : str
        Path to the downloaded file

    """
    if sabdab:
        try:
            url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/pdb/{pdb_id}/?scheme=chothia"
            local_path = os.path.join(local_folder, f"{pdb_id}.pdb")
            _download_file(url, local_path)
            return local_path
        except BaseException:
            raise RuntimeError(f"Could not download {pdb_id}")
    if "-" in pdb_id:
        pdb_id, biounit = pdb_id.split("-")
        filenames = {
            "cif": f"{pdb_id}-assembly{biounit}.cif.gz",
            "pdb": f"{pdb_id}.pdb{biounit}.gz",
        }
        local_name = f"{pdb_id}-{biounit}"
    else:
        filenames = {
            "cif": f"{pdb_id}.cif.gz",
            "pdb": f"{pdb_id}.pdb.gz",
        }
        local_name = pdb_id
    for t in filenames:
        local_path = os.path.join(local_folder, local_name + f".{t}.gz")
        try:
            url = f"https://files.rcsb.org/download/{filenames[t]}"
            _download_file(url, local_path)
            return local_path
        except BaseException:
            pass
    raise RuntimeError(f"Could not download {pdb_id}")


def download_fasta(pdb_id, local_folder="."):
    """
    Download a FASTA file from the RCSB PDB database.

    Parameters
    ----------
    pdb_id : str
        PDB ID of the protein to download
    local_folder : str, default "."
        Folder to save the downloaded file to

    Returns
    -------
    local_path : str
        Path to the downloaded file

    """
    if "-" in pdb_id:
        pdb_id = pdb_id.split("-")[0]
    downloadurl = "https://www.rcsb.org/fasta/entry/"
    pdbfn = pdb_id + "/download"
    local_path = os.path.join(local_folder, f"{pdb_id.lower()}.fasta")

    url = downloadurl + pdbfn
    urllib.request.urlretrieve(url, local_path)
    return local_path


def get_pdb_ids(
    resolution_thr=3.5,
    pdb_snapshot=None,
    filter_methods=True,
):
    """Get PDB ids from PDB API."""
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


def _download_pdb(pdb_id, local_folder):
    try:
        return download_pdb(pdb_id, local_folder)
    except RuntimeError:
        return pdb_id


def _download_fasta(pdb_id, local_folder):
    try:
        return download_fasta(pdb_id, local_folder)
    except RuntimeError:
        return pdb_id


def download_filtered_pdb_files(
    resolution_thr=3.5,
    pdb_snapshot=None,
    filter_methods=True,
    n=None,
    local_folder=".",
    load_live=False,
):
    """Download filtered PDB files and return a list of local file paths."""
    ordered_folders, pdb_ids = get_pdb_ids(
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
        pdbs = {x.split("-")[0] for x in ids}
        future_to_key = {
            executor.submit(
                lambda x: _download_fasta(x, datadir=local_folder), key
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
        pdb_ids=ids, tmp_folder=local_folder, snapshots=[ordered_folders[0]]
    )
    paths = [item for sublist in paths for item in sublist]
    error_ids = [x for x in paths if not x.endswith(".gz")]
    paths = [x for x in paths if x.endswith(".gz")]
    if load_live:
        print("Downloading newest structure files...")
        live_paths = p_map(
            lambda x: _download_pdb(x, tmp_folder=local_folder), error_ids
        )
        error_ids = []
        for x in live_paths:
            if x.endswith(".gz"):
                paths.append(x)
            else:
                error_ids.append(x)
    return paths, error_ids


def _download_sabdab_by_method(
    methods,
    resolution_thr=3.5,
    local_folder=".",
):
    for method in methods:
        html = _make_sabdab_html(method, resolution_thr)
        page = requests.get(html)
        soup = BeautifulSoup(page.text, "html.parser")
        try:
            zip_ref = soup.find_all(
                lambda t: t.name == "a" and t.text.startswith("zip")
            )[0]["href"]
            zip_ref = "https://opig.stats.ox.ac.uk" + zip_ref
        except BaseException:
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
                os.path.join(local_folder, f"pdb_{'_'.join(method)}.zip"),
            ]
        )
        if (
            os.stat(os.path.join(local_folder, f"pdb_{'_'.join(method)}.zip")).st_size
            == 0
        ):
            raise RuntimeError("The archive was not downloaded")


def _download_sabdab_all(
    local_folder=".",
):
    print("Trying to download all data...")
    data_html = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/"
    index_html = "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/"
    subprocess.run(
        [
            "wget",
            data_html,
            "-O",
            os.path.join(local_folder, "pdb_all.zip"),
        ]
    )
    if os.stat(os.path.join(local_folder, "pdb_all.zip")).st_size == 0:
        raise RuntimeError("The archive was not downloaded")
    subprocess.run(
        [
            "unzip",
            os.path.join(local_folder, "pdb_all.zip"),
            "-d",
            local_folder,
        ]
    )
    subprocess.run(
        [
            "wget",
            index_html,
            "-O",
            os.path.join(local_folder, "all_structures", "summary.tsv"),
        ]
    )
    if (
        os.stat(os.path.join(local_folder, "all_structures", "summary.tsv")).st_size
        == 0
    ):
        raise RuntimeError("The index was not downloaded")


def download_filtered_sabdab_files(
    resolution_thr=3.5,
    filter_methods=True,
    pdb_snapshot=None,
    local_folder=".",
    sabdab_data_path=None,
    require_antigen=True,
    n=None,
):
    """Download filtered SAbDab files and return a list of local file paths."""
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    if pdb_snapshot is not None:
        pdb_snapshot = datetime.strptime(pdb_snapshot, "%Y%m%d")
    if filter_methods:
        methods = ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"]
    else:
        methods = ["All"]
    methods = [x.split() for x in methods]
    if sabdab_data_path is None:
        try:
            _download_sabdab_by_method(
                methods=methods, resolution_thr=resolution_thr, tmp_folder=local_folder
            )
            paths = [
                os.path.join(local_folder, f"pdb_{'_'.join(method)}.zip")
                for method in methods
            ]
        except RuntimeError:
            _download_sabdab_all(tmp_folder=local_folder)
            paths = [os.path.join(local_folder, "all_structures")]
            sabdab_data_path = os.path.join(local_folder, "all_structures")
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
                    except zipfile.error:
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
                    shutil.move(pdb_path, os.path.join(local_folder, f"{id}.pdb"))
                else:
                    shutil.copy(pdb_path, os.path.join(local_folder, f"{id}.pdb"))
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
                lambda x: _download_fasta(x, datadir=local_folder), key
            ): key
            for key in pdb_ids
        }
        _ = [
            x.result()
            for x in tqdm(futures.as_completed(future_to_key), total=len(pdb_ids))
        ]
    paths = [(os.path.join(local_folder, f"{x[0]}.pdb"), x[1]) for x in ids]
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
    """Download filtered structure files and return a list of local file paths."""
    if sabdab:
        out = download_filtered_sabdab_files(
            resolution_thr=resolution_thr,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            tmp_folder=tmp_folder,
            sabdab_data_path=sabdab_data_path,
            require_antigen=require_antigen,
            n=n,
        )
    else:
        out = download_filtered_pdb_files(
            resolution_thr=resolution_thr,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            tmp_folder=tmp_folder,
            load_live=load_live,
            n=n,
        )
    return out
