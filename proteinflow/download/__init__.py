"""Functions for downloading protein data from various sources."""

import multiprocessing
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
from joblib import Parallel, delayed
from p_tqdm import p_map
from rcsbsearch import Attr
from tqdm import tqdm

from proteinflow.constants import ALLOWED_AG_TYPES
from proteinflow.download.boto import (
    _download_dataset_dicts_from_s3,
    _download_dataset_from_s3,
    _download_s3_parallel,
    _get_s3_paths_from_tag,
    _s3list,
)


def _download_file(url, local_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url)
    open(local_path, "wb").write(response.content)


def download_pdb(pdb_id, local_folder=".", sabdab=False):
    """Download a PDB file from the RCSB PDB database.

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
    pdb_id = pdb_id.lower()
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
    pdb_id = pdb_id.lower()
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
    max_chains=5,
    pdb_id_list_path=None,
):
    """Get PDB ids from PDB API."""
    if pdb_id_list_path is not None:
        pdb_ids = []  # List to store the extracted elements

        try:
            with open(pdb_id_list_path) as file:
                # Read lines from the file
                lines = file.readlines()

                # Process each line
                for line in lines:
                    # Extract elements from the line (example: splitting by whitespace)
                    line_elements = line.split()

                    # Add extracted elements to the list
                    pdb_ids.extend(line_elements)
                pdb_ids = np.unique(pdb_ids)
        except FileNotFoundError:
            print(f"The file '{pdb_id_list_path}' does not exist.")
    else:
        # get filtered PDB ids from PDB API
        pdb_ids = (
            Attr("rcsb_entry_info.selected_polymer_entity_types")
            .__eq__("Protein (only)")
            .or_("rcsb_entry_info.polymer_composition")
            .__eq__("protein/oligosaccharide")
        )
        # if include_na:
        #     pdb_ids = pdb_ids.or_('rcsb_entry_info.polymer_composition').in_(["protein/NA", "protein/NA/oligosaccharide"])

        if max_chains is not None:
            pdb_ids = pdb_ids.and_(
                "rcsb_assembly_info.polymer_entity_instance_count_protein"
            ).__le__(max_chains)
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
    """Download a PDB file and return a local path or the PDB ID if download failed."""
    try:
        return download_pdb(pdb_id, local_folder)
    except RuntimeError:
        return pdb_id


def _download_fasta(pdb_id, local_folder):
    """Download a FASTA file and return a local path or the PDB ID if download failed."""
    try:
        return download_fasta(pdb_id, local_folder)
    except Exception:
        return pdb_id


def download_filtered_pdb_files(
    resolution_thr=3.5,
    pdb_snapshot=None,
    filter_methods=True,
    n=None,
    local_folder=".",
    load_live=False,
    max_chains=5,
    pdb_id_list_path=None,
):
    """Download filtered PDB files and return a list of local file paths.

    Parameters
    ----------
    resolution_thr : float, default 3.5
        Resolution threshold
    pdb_snapshot : str, default None
        PDB snapshot to download from
    filter_methods : bool, default True
        Whether to filter by experimental method
    n : int, default None
        Number of PDB files to download (for debugging)
    local_folder : str, default "."
        Folder to save the downloaded files to
    load_live : bool, default False
        Whether to load the PDB files from the RCSB PDB database directly
        instead of downloading them from the PDB snapshots
    max_chains : int, default 5
        Maximum number of chains per biounit
    pdb_id_list_path : str, default None
        Path to a file with a list of PDB IDs to download

    Returns
    -------
    local_paths : list of str
        List of local file paths
    error_ids : list of str
        List of PDB IDs that could not be downloaded
    """
    ordered_folders, pdb_ids = get_pdb_ids(
        resolution_thr=resolution_thr,
        pdb_snapshot=pdb_snapshot,
        filter_methods=filter_methods,
        max_chains=max_chains,
        pdb_id_list_path=pdb_id_list_path,
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
                lambda x: _download_fasta(x, local_folder=local_folder), key
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
    """Download SAbDab files by method.

    Parameters
    ----------
    methods : list of str
        List of methods to download
    resolution_thr : float, default 3.5
        Resolution threshold
    local_folder : str, default "."
        Folder to save the downloaded files to

    Returns
    -------
    local_paths : list of str
        List of local file paths
    error_ids : list of str
        List of PDB IDs that could not be downloaded

    """
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
    """Download all SAbDab files.

    Parameters
    ----------
    local_folder : str, default "."
        Folder to save the downloaded files to

    Returns
    -------
    local_paths : list of str
        List of local file paths
    error_ids : list of str
        List of PDB IDs that could not be downloaded

    """
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
    """Download filtered SAbDab files and return a list of local file paths.

    Parameters
    ----------
    resolution_thr : float, default 3.5
        Resolution threshold
    filter_methods : bool, default True
        Whether to filter by method
    pdb_snapshot : str, default None
        PDB snapshot date in YYYYMMDD format
    local_folder : str, default "."
        Folder to save the downloaded files to
    sabdab_data_path : str, default None
        Path to the SAbDab data folder
    require_antigen : bool, default True
        Whether to require the presence of an antigen
    n : int, default None
        Number of structures to download (for debugging)

    Returns
    -------
    local_paths : list of str
        List of local file paths
    error_ids : list of str
        List of PDB IDs that could not be downloaded

    """
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
                methods=methods,
                resolution_thr=resolution_thr,
                local_folder=local_folder,
            )
            paths = [
                os.path.join(local_folder, f"pdb_{'_'.join(method)}.zip")
                for method in methods
            ]
        except RuntimeError:
            _download_sabdab_all(local_folder=local_folder)
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
                lambda x: _download_fasta(x, local_folder=local_folder), key
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
    local_folder=".",
    load_live=False,
    sabdab=False,
    sabdab_data_path=None,
    require_antigen=False,
    max_chains=5,
    pdb_id_list_path=None,
):
    """Download filtered structure files and return a list of local file paths."""
    if sabdab:
        paths, error_ids = download_filtered_sabdab_files(
            resolution_thr=resolution_thr,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            local_folder=local_folder,
            sabdab_data_path=sabdab_data_path,
            require_antigen=require_antigen,
            n=n,
        )
    else:
        paths, error_ids = download_filtered_pdb_files(
            resolution_thr=resolution_thr,
            filter_methods=filter_methods,
            pdb_snapshot=pdb_snapshot,
            local_folder=local_folder,
            load_live=load_live,
            n=n,
            max_chains=max_chains,
            pdb_id_list_path=pdb_id_list_path,
        )
    paths = [(x, _get_fasta_path(x)) for x in paths]
    return paths, error_ids


def _make_sabdab_html(method, resolution_thr):
    """Make a URL for SAbDab search."""
    html = f"https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?ABtype=All&method={'+'.join(method)}&species=All&resolution={resolution_thr}&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=All&isin_covabdab=All&isin_therasabdab=All&chothiapos=&restype=ALA&field_0=Antigens&keyword_0=#downloads"
    return html


def _get_fasta_path(pdb_path):
    """Get the path to the fasta file corresponding to the pdb file."""
    if isinstance(pdb_path, tuple):
        pdb_path = pdb_path[0]
    pdb_id = os.path.basename(pdb_path).split(".")[0].split("-")[0]
    return os.path.join(os.path.dirname(pdb_path), f"{pdb_id}.fasta")


def _download_dataset(tag, local_datasets_folder="./data/"):
    """Download the pre-processed data and the split dictionaries.

    Parameters
    ----------
    tag : str
        name of the dataset (check `get_available_tags` to see the options)
    local_dataset_folder : str, default "./data/"
        the local folder that will contain proteinflow dataset folders, temporary files and logs

    Returns
    -------
    data_folder : str
        the path to the downloaded data folder

    """
    s3_data_path, s3_dict_path = _get_s3_paths_from_tag(tag)
    data_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")
    dict_folder = os.path.join(
        local_datasets_folder, f"proteinflow_{tag}", "splits_dict"
    )

    print("Downloading dictionaries for splitting the dataset...")
    _download_dataset_dicts_from_s3(dict_folder, s3_dict_path)
    print("Done!")

    _download_dataset_from_s3(dataset_path=data_folder, s3_path=s3_data_path)
    return data_folder


def _create_jobs(file_path, strings, results):
    """Create jobs for parallel processing."""
    # Perform your job creation logic here
    jobs = []
    for string in strings:
        for i in range(results[string]):
            jobs.append((file_path, string, i))
    return jobs


def _process_strings(strings):
    """Process strings in parallel."""
    results = {}
    processed_results = Parallel(n_jobs=-1)(
        delayed(_get_number_of_chains)(string) for string in strings
    )

    for string, result in zip(strings, processed_results):
        results[string] = result

    return results


def _write_string_to_file(file_path, string, i):
    """Write a string to a file."""
    with open(file_path, "a") as file:
        file.write(string.upper() + "-" + str(i + 1) + "\n")


def _parallel_write_to_file(file_path, jobs):
    """Write a list of strings to a file in parallel."""
    # Create a multiprocessing Pool with the desired number of processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Map the write_string_to_file function to each string in the list
    pool.starmap(_write_string_to_file, jobs)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    print(f"The list has been written to the file '{file_path}' successfully.")


def _write_list_to_file(file_path, string_list):
    """Write a list of strings to a file."""
    try:
        with open(file_path, "w") as file:
            # Write each string in the list to the file
            for string in string_list:
                file.write(string + "\n")  # Add a newline character after each string

        print(f"The list has been written to the file '{file_path}' successfully.")

    except OSError:
        print(f"An error occurred while writing to the file '{file_path}'.")


def _get_number_of_chains(pdb_id):
    """Return the number of chains in a PDB file."""
    api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Extracting chain IDs
        chains = set()
        if "rcsb_entry_container_identifiers" in data:
            entity_container_identifiers = data["rcsb_entry_container_identifiers"]
            if "assembly_ids" in entity_container_identifiers:
                return len(entity_container_identifiers["assembly_ids"])

        num_chains = len(chains)

        return num_chains

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return 0


def _get_chain_pdb_ids(pdb_id_list_path, tmp_folder):
    print("Generating chain pdb ids")
    pdb_ids = []
    with open(pdb_id_list_path) as file:
        # Read lines from the file
        lines = file.readlines()

        # Process each line
        for line in lines:
            # Extract elements from the line (example: splitting by whitespace)
            line_elements = line.split()

            # Add extracted elements to the list
            pdb_ids.extend(line_elements)
    results = _process_strings(pdb_ids)
    new_file_path = tmp_folder + "chain_id_" + pdb_id_list_path.split("/")[-1]
    jobs = _create_jobs(new_file_path, pdb_ids, results)
    _parallel_write_to_file(new_file_path, jobs)
    return new_file_path
