"""
ProteinFlow is an open-source Python library that streamlines the pre-processing of protein structure data for deep learning applications. ProteinFlow enables users to efficiently filter, cluster, and generate new datasets from resources like the Protein Data Bank (PDB) and SAbDab (The Structural Antibody Database).

Here are some of the key features we currently support:

- ⛓️ Processing of both single-chain and multi-chain protein structures (Biounit PDB definition)
- 🏷️ Various featurization options can be computed, including secondary structure features, torsion angles, etc.
- 💾 A variety of data loading options and conversions to cater to different downstream training frameworks
- 🧬 Access to up-to-date, pre-computed protein structure datasets

![overview](https://raw.githubusercontent.com/adaptyvbio/ProteinFlow/main/media/pf-1.png)

---

## Installation
conda:
```bash
# This should take a few minutes, be patient
conda install -c conda-forge -c bioconda -c adaptyvbio proteinflow
```

pip:
```bash
pip install proteinflow
```

docker:
```bash
docker pull adaptyvbio/proteinflow
```

### Troubleshooting
- If you are using python 3.10 and encountering installation problems, try running `python -m pip install prody==2.4.0` before installing `proteinflow`.
- If you are planning to generate new datasets and installed `proteinflow` with `pip`, you will need to additionally install [`mmseqs`](https://github.com/soedinglab/MMseqs2).
- Generating new datasets also depends on the `rcsbsearch` package and the latest release [v0.2.3](https://github.com/sbliven/rcsbsearch/releases/tag/v0.2.3) is currently not working correctly. The recommended fix is installing the version from [this pull request](https://github.com/sbliven/rcsbsearch/pull/6).
```bash
python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
```
- The docker image can be accessed in interactive mode with this command.
```bash
docker run -it -v /path/to/data:/media adaptyvbio/proteinflow bash
```

## Usage
### Downloading pre-computed datasets (stable)
Already precomputed datasets with consensus set of parameters and can be accessed and downloaded using the `proteinflow`. package. Check the output of `proteinflow check_tags` for a list of available tags.
```bash
proteinflow download --tag 20230102_stable
```

### Running the pipeline (PDB)
You can also run `proteinflow` with your own parameters. Check the output of `proteinflow check_snapshots` for a list of available PDB snapshots (naming rule: `yyyymmdd`).

For instance, let's generate a dataset with the following description:
- resolution threshold: 5 angstrom,
- PDB snapshot: 20190101,
- structure methods accepted: all (x-ray christolography, NRM, Cryo-EM),
- sequence identity threshold for clustering: 40% sequence similarity,
- maximum length per sequence: 1000 residues,
- minimum length per sequence: 5 residues,
- maximum fraction of missing values at the ends: 10%,
- size of validation subset: 10%.

```bash
proteinflow generate --tag new --resolution_thr 5 --pdb_snapshot 20190101 --not_filter_methods --min_seq_id 0.4 --max_length 1000 --min_length 5 --missing_ends_thr 0.1 --valid_split 0.1
```
See the [docs](https://adaptyvbio.github.io/ProteinFlow/) (or `proteinflow generate --help`) for the full list of parameters and more information.

A registry of all the files that are removed during the filtering as well as description with the reason for their removal is created automatically for each `generate` command. The log files are save (at `data/logs` by default) and a summary can be accessed running `proteinflow get_summary {log_path}`.

### Running the pipeline (SAbDab)
You can also use the `--sabdab` option in `proteinflow generate` to load files from SAbDab and cluster them based on CDRs. By default the `--sabdab` tag will download the latest up-to-date version of the SabDab dataset and cluster the antibodies based on their CDR sequence.
Alternatively, it can be used together with the tag `--sabdab_data_path` to process a custom SAbDab-like zip file or folder. This allows you to use search and query tools from the [SabDab web interface](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/) to create a custom dataset by downloading the archived zip file of the structures selected. (Under Downloads section of your SabDab query).

SAbDab sequences clustering is done across all 6 Complementary Determining Regions (CDRs) - CDRH1, CDRH2, CDRH3, CDRL1, CDRL2, CDRL3, based on the [Chothia numbering](https://pubmed.ncbi.nlm.nih.gov/9367782/) implemented by SabDab. CDRs from nanobodies and other synthetic constructs are clustered together with other heavy chain CDRs. The resulting CDR clusters are split into training, test and validation in a way that ensures that every PDB file only appears in one subset.

For instance, let's generate a dataset with the following description:
- SabDab version: latest (up-to-date),
- resolution threshold: 5 angstrom,
- structure methods accepted: all (x-ray christolography, NRM, Cryo-EM),
- sequence identity threshold for clustering (CDRs): 40%,
- size of validation subset: 10%.

```bash
proteinflow generate --sabdab --resolution_thr 5 --not_filter_methods --min_seq_id 0.4 --valid_split 0.1
```
### Splitting
By default, both `proteinflow generate` and `proteinflow download` will also split your data into training, test and validation according to MMseqs2 clustering and homomer/heteromer/single chain proportions. However, you can skip this step with a `--skip_splitting` flag and then perform it separately with the `proteinflow split` command.

The following command will perform the splitting with a 10% validation set, a 5% test set and a 50% threshold for sequence identity clusters.
```bash
proteinflow split --tag new --valid_split 0.1 --test_split 0.5 --min_seq_id 0.5
```

Use the `--exclude_chains` and `--exclude_threshold` parameters to move all biounits that contain chains similar to what you specify to a separate folder.

### Using the data
The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are the following:
- `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
- `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (check `proteinflow.sidechain_order()` for the order of atoms),
- `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
    zeros to missing values,
- `'seq'`: a string of length `L` with residue types.

In a SAbDab datasets, an additional key is added to the dictionary:
- `'cdr'`: a `'numpy'` array of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
    and non-CDR residues are marked with `'-'`.

Once your data is ready, you can open the files with `pickle`.

```python
import pickle
import os

train_folder = "./data/proteinflow_new/training"
for filename in os.listdir(train_folder):
    with open(os.path.join(train_folder, filename), "rb") as f:
        data = pickle.load(f)
    crd_bb = data["crd_bb"]
    seq = data["seq"]
    ...
```

Alternatively, you can use our `ProteinDataset` or `ProteinLoader` classes
for convenient processing. Among other things, they allow for feature extraction, single chain / homomer / heteromer filtering and randomized sampling from sequence identity clusters.

For example, here is how we can create a data loader that:
- samples a different cluster representative at every epoch,
- extracts dihedral angles, sidechain orientation and secondary structure features,
- only loads pairs of interacting proteins (larger biounits are broken up into pairs),
- has batch size 8.

```python
from proteinflow import ProteinLoader
train_loader = ProteinLoader.from_args(
    "./data/proteinflow_new/training",
    clustering_dict_path="./data/proteinflow_new/splits_dict/train.pickle",
    node_features_type="dihedral+sidechain_orientation+secondary_structure",
    entry_type="pair",
    batch_size=8,
)
for batch in train_loader:
    crd_bb = batch["X"] # (B, L, 4, 3)
    seq = batch["S"] # (B, L)
    sse = batch["secondary_structure"] # (B, L, 3)
    to_predict = batch["masked_res"] # (B, L), 1 where the residues should be masked, 0 otherwise
    ...
```
"""
__pdoc__ = {
    "utils": False,
    "scripts": False,
    "constants": False,
    "custom_mmcif": False,
    "pdb": False,
    "sequences": False,
}
__docformat__ = "numpy"

import os
import pickle
import random
import shutil
import string
import subprocess
import tempfile
import urllib
import urllib.request
import warnings
import zipfile
from collections import defaultdict
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import boto3
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.config import Config
from bs4 import BeautifulSoup
from editdistance import eval as edit_distance
from p_tqdm import p_map
from rcsbsearch import Attr
from tqdm import tqdm

from proteinflow.constants import ALLOWED_AG_TYPES, SIDECHAIN_ORDER
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
    _log_exception,
    _log_removed,
    _make_sabdab_html,
    _raise_rcsbsearch,
)


def _get_split_dictionaries(
    tmp_folder="./data/tmp_pdb",
    output_folder="./data/pdb",
    split_tolerance=0.2,
    test_split=0.05,
    valid_split=0.05,
    out_split_dict_folder="./data/dataset_splits_dict",
    min_seq_id=0.3,
):
    """Split preprocessed data into training, validation and test.

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

    classes_dict = train_classes_dict
    for d in [valid_classes_dict, test_classes_dict]:
        for k, v in d.items():
            classes_dict[k].update(v)

    with open(os.path.join(out_split_dict_folder, "classes.pickle"), "wb") as f:
        pickle.dump(classes_dict, f)
    with open(os.path.join(out_split_dict_folder, "train.pickle"), "wb") as f:
        pickle.dump(train_clusters_dict, f)
    with open(os.path.join(out_split_dict_folder, "valid.pickle"), "wb") as f:
        pickle.dump(valid_clusters_dict, f)
    with open(os.path.join(out_split_dict_folder, "test.pickle"), "wb") as f:
        pickle.dump(test_clusters_dict, f)


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


def _get_pdb_ids(
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


def _download_live(id, tmp_folder):
    """Download a PDB file from the PDB website."""
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
        except BaseException:
            pass
    return id


def _download_fasta_f(pdb_id, datadir):
    """Download a fasta file from the PDB website."""
    downloadurl = "https://www.rcsb.org/fasta/entry/"
    pdbfn = pdb_id + "/download"
    outfnm = os.path.join(datadir, f"{pdb_id.lower()}.fasta")

    url = downloadurl + pdbfn
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm

    except Exception:
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
    """Download filtered PDB files and return a list of local file paths."""
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
        pdbs = {x.split("-")[0] for x in ids}
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


def _load_sabdab(
    resolution_thr=3.5,
    filter_methods=True,
    pdb_snapshot=None,
    tmp_folder="data/tmp",
    sabdab_data_path=None,
    require_antigen=True,
    n=None,
):
    """Download filtered SAbDab files and return a list of local file paths."""
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
    """Download filtered structure files and return a list of local file paths."""
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
    """Download a pre-computed dataset with train/test/validation splits.

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
    random_seed=42,
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
    random_seed : int, default 42
        the random seed to use for splitting

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
    tmp_folder = os.path.join(tempfile.gettempdir(), tag + tmp_id)
    os.makedirs(tmp_folder)
    output_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")

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
            random_seed=random_seed,
        )
    shutil.rmtree(tmp_folder)
    return log_dict


def _get_excluded_files(
    tag, local_datasets_folder, tmp_folder, exclude_chains, exclude_threshold
):
    """Get a list of files to exclude from the dataset.

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
    random_seed=42,
):
    """Split `proteinflow` entry files into training, test and validation.

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
    random_seed : int, default 42
        random seed for reproducibility (set to `None` to use a random seed)

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
        random.seed(random_seed)
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
    """Move files from train, test, validation and excluded folders back into the main folder.

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
    """Get a dictionary of sidechain atom orders.

    Returns
    -------
    order_dict : dict
        a dictionary where keys are 3-letter aminoacid codes and values are lists of atom names (in PDB format) that correspond to
        coordinates in the `'crd_sc'` array generated by the `run_processing` function

    """
    return SIDECHAIN_ORDER


def get_error_summary(log_file, verbose=True):
    """Get an exception summary.

    The output is a dictionary where keys are recognized exception names and values are lists of
    PDB ids that caused the exceptions.

    Parameters
    ----------
    log_file : str
        the log file path
    verbose : bool, default True
        if `True`, the statistics are written in the standard output

    Returns
    -------
    log_dict : dict
        a dictionary where keys are recognized exception names and values are lists of PDB ids that
        caused the exceptions

    """
    stats = defaultdict(lambda: [])
    with open(log_file) as f:
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
    """Get a list of PDB snapshots available for downloading.

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
    """Get a list of tags available for downloading.

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
