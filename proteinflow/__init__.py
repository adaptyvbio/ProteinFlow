"""
Proteinflow is an open-source Python library that streamlines the pre-processing of protein structure data for deep learning applications. ProteinFlow enables users to efficiently filter, cluster, and generate new datasets from resources like the Protein Data Bank (PDB).

Here are some of the key features we currently support:

- ⛓️ Processing of both single-chain and multi-chain protein structures (Biounit PDB definition)
- 🏷️ Various featurization options can be computed, including secondary structure features, torsion angles, etc.
- 💾 A variety of data loading options and conversions to cater to different downstream training frameworks
- 🧬 Access to up-to-date, pre-computed protein structure datasets

![overview](https://raw.githubusercontent.com/adaptyvbio/ProteinFlow/main/media/pf-1.png)

---

## Installation
Recommended: create a new `conda` environment and install `proteinflow` with `pip`. 
```bash
conda create --name proteinflow -y
conda activate proteinflow
python -m pip install proteinflow
```

If you are using `python 3.10` and encountering installation problems, try running `python -m pip install prody==2.4.0` before installing `proteinflow`.

### Additional requirements
In most cases, running the commands is enough. However, if you are planning to generate a new dataset, there is a couple additional requirements.

First, you will need to install `mmseqs`. The recommended way is to run the following command in your `conda` environment but there are alternative methods you can see [here](https://github.com/soedinglab/MMseqs2).
```bash
conda install -y -c conda-forge -c bioconda mmseqs2
```

In addition, `proteinflow` depends on the `rcsbsearch` package and the latest release [v0.2.3](https://github.com/sbliven/rcsbsearch/releases/tag/v0.2.3) is currently not working correctly. Follow the recommended fix:
```bash
python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
```

Finally, you can use our [docker image](https://hub.docker.com/r/adaptyvbio/proteinflow/tags) as an alternative.
```bash
docker run -it -v /path/to/data:/media adaptyvbio/proteinflow bash
```

## Usage
### Downloading pre-computed datasets
We have already run the pipeline with a consensus set of parameters and saved the results at a server. You can download the resulting dataset with `proteinflow`. Check the output of `proteinflow check_tags` for a list of available tags.
```bash
proteinflow download --tag paper 
```

See `proteinflow.download_data` (or run `proteinflow download --help`) for more information.

### Running the pipeline
You can also run `proteinflow` with your own parameters. Check the output of `proteinflow check_snapshots` for a list of available snapshots (naming rule: `{year}{month}{day}`).

For instance, let's generate a dataset with the following description:
- resolution threshold 5 $\AA$,
- PDB snapshot 20190101,
- all structure methods accepted,
- sequence identity threshold for clustering 40%,
- maximum length per sequence 1000 residues,
- minimum length per sequence 5 residues,
- maximum fraction of missing values at the ends 10%,
- validation subset 10%.

```bash
proteinflow generate --tag new --resolution_thr 5 --pdb_snapshot 20190101 --not_filter_methods --min_seq_id 0.4 --max_length 1000 --min_length 5 --missing_ends_thr 0.1 --valid_split 0.1
```
See `proteinflow.generate_data` (or run `proteinflow generate --help`) for the full list of parameters and more information.

The reasons for filtering files out are logged in text files (at `data/logs` by default). To get a summary, run `proteinflow get_summary {log_path}`.

### Splitting
By default, both `proteinflow generate` and `proteinflow download` will also split your data into training, test and validation according to MMseqs2 clustering and homomer/heteromer/single chain proportions. However, you can skip this step with a `--skip_splitting` flag and then perform it separately with the `proteinflow split` command.

The following command will perform the splitting with a 10% validation set, a 5% test set and a 50% threshold for sequence identity clusters.
```bash
proteinflow split --tag new --valid_split 0.1 --test_split 0.5 --min_seq_id 0.5
```

See `proteinflow.split_data` (or run `proteinflow split --help`) for more information.

### Using the data
The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are the following:

- `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
- `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (check `proteinflow.sidechain_order()` for the order of atoms),
- `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
    zeros to missing values,
- `'seq'`: a string of length `L` with residue types.

Once your data is ready, you can open the files directly with `pickle` to access this data.

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
- generates a geometric mask for 10% of one of the chains (random generation at every pass),
- has batch size 8.

```python
from proteinflow import ProteinLoader
train_loader = ProteinLoader.from_args(
    "./data/proteinflow_new/training", 
    clustering_dict_path="./data/proteinflow_new/splits_dict/train.pickle",
    node_features_type="dihedral+sidechain_orientation+secondary_structure",
    entry_type="pair",
    mask_frac=0.1,
    batch_size=8,
)
for batch in train_loader:
    crd_bb = batch["X"] #(B, L, 4, 3)
    seq = batch["S"] #(B, L)
    sse = batch["secondary_structure"] #(B, L, 3)
    to_predict = batch["masked_res"] #(B, L), 1 where the residues should be masked, 0 otherwise
    ...
```

See `proteinflow.ProteinLoader` for more information.

"""

__pdoc__ = {"utils": False, "scripts": False}
__docformat__ = "numpy"

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

REVERSE_D3TO1 = {v: k for k, v in D3TO1.items()}
REVERSE_D3TO1["X"] = "GLY"

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
    if "404 Client Error" in str(e):
        raise RuntimeError(
            'Quering rcsbsearch is failing. Please install a version of rcsbsearch where this error is solved:\npython -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"'
        )
    else:
        raise e


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
    load_live=False,
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
        the PDB snapshot to use, by default the latest is used
    load_live : bool, default False
        if `True`, load the files that are not in the latest PDB snapshot from the PDB FTP server (forced to `False` if `pdb_snapshot` is not `None`)

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
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if pdb_snapshot is not None:
        load_live = False

    i = 0
    while os.path.exists(os.path.join(log_folder, f"log_{i}.txt")):
        i += 1
    LOG_FILE = os.path.join(log_folder, f"log_{i}.txt")
    print(f"Log file: {LOG_FILE} \n")
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n\n"
    with open(LOG_FILE, "a") as f:
        f.write(date_time)
        if tag is not None:
            f.write(f"tag: {tag} \n\n")

    # get filtered PDB ids from PDB API
    pdb_ids = (
        Attr("rcsb_entry_info.selected_polymer_entity_types")
        .__eq__("Protein (only)")
        .or_("rcsb_entry_info.polymer_composition")
        .__eq__("protein/oligosaccharide")
    )
    # if include_na:
    #     pdb_ids = pdb_ids.or_('rcsb_entry_info.polymer_composition').in_(["protein/NA", "protein/NA/oligosaccharide"])

    if RESOLUTION_THR is not None:
        pdb_ids = pdb_ids.and_("rcsb_entry_info.resolution_combined").__le__(
            RESOLUTION_THR
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

    # session = boto3.session.Session()
    # s3_client = session.client("s3", config=Config(signature_version=UNSIGNED))

    session = get_session()

    def download_live(id):
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

    def download_fasta_f(pdb_id, datadir):
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

    def process_f(
        local_path,
        show_error=False,
        force=True,
    ):
        try:
            # local_path = download_f(pdb_id, s3_client=s3_client, load_live=load_live)
            fn = os.path.basename(local_path)
            pdb_id = fn.split(".")[0]
            target_file = os.path.join(OUTPUT_FOLDER, pdb_id + ".pickle")
            if not force and os.path.exists(target_file):
                raise PDBError("File already exists")
            pdb_dict = _open_structure(
                local_path,
                tmp_folder=TMP_FOLDER,
            )
            # filter and convert
            pdb_dict = _align_structure(
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

    # for x in ["1a52-3", "1a52-4", "1a52-2", "1a52-1"]:
    #     process_f(x, show_error=True, force=force)

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            print("Get a file list...")
            ids = []
            for i, x in enumerate(tqdm(pdb_ids)):
                ids.append(x)
                if n is not None and i == n:
                    break
            print("Download fasta files...")
            pdbs = set([x.split("-")[0] for x in ids])
            future_to_key = {
                executor.submit(
                    lambda x: download_fasta_f(x, datadir=tmp_folder), key
                ): key
                for key in pdbs
            }
            _ = [
                x.result()
                for x in tqdm(futures.as_completed(future_to_key), total=len(pdbs))
            ]

        # _ = [process_f(x, force=force, load_live=load_live) for x in tqdm(ids)]
        print("Download structure files...")
        paths = _download_s3_parallel(
            pdb_ids=ids, tmp_folder=tmp_folder, snapshots=[ordered_folders[0]]
        )
        paths = [item for sublist in paths for item in sublist]
        error_ids = [x for x in paths if not x.endswith(".gz")]
        paths = [x for x in paths if x.endswith(".gz")]
        if load_live:
            print("Download newest structure files...")
            live_paths = p_map(download_live, error_ids)
            error_ids = []
            for x in live_paths:
                if x.endswith(".gz"):
                    paths.append(x)
                else:
                    error_ids.append(x)
        for id in error_ids:
            with open(LOG_FILE, "a") as f:
                f.write(f"<<< Could not download PDB/mmCIF file: {id} \n")
        print("Filter and process...")
        _ = p_map(lambda x: process_f(x, force=force), paths)
    except Exception as e:
        _raise_rcsbsearch(e)

    stats = get_error_summary(LOG_FILE, verbose=False)
    not_found_error = "<<< PDB / mmCIF file downloaded but not found"
    while not_found_error in stats:
        with open(LOG_FILE, "r") as f:
            lines = [x for x in f.readlines() if not x.startswith(not_found_error)]
        os.remove(LOG_FILE)
        with open(f"{LOG_FILE}_tmp", "a") as f:
            for line in lines:
                f.write(line)
        _ = p_map(lambda x: process_f(x, force=force), stats[not_found_error])
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


class _PadCollate:
    """
    A variant of `collate_fn` that pads according to the longest sequence in
    a batch of sequences

    If `mask_residues` is `True`, an additional `'masked_res'` key is added to the output. The value is a binary
    tensor where 1 denotes the part that needs to be predicted and 0 is everything else. The tensors are generated
    according to the following rules:
    - if `mask_whole_chains` is `True`, the whole chain is masked
    - if `mask_frac` is given, the number of residues to mask is `mask_frac` times the length of the chain,
    - otherwise, the number of residues to mask is sampled uniformly from the range [`lower_limit`, `upper_limit`].

    If `force_binding_sites_frac` > 0 and `mask_whole_chains` is `False`, in the fraction of cases where a chain
    from a polymer is sampled, the center of the masked region will be forced to be in a binding site.
    """

    def __init__(
        self,
        mask_residues=True,
        lower_limit=15,
        upper_limit=100,
        mask_frac=None,
        mask_whole_chains=False,
        force_binding_sites_frac=0.15,
    ):
        """
        Parameters
        ----------
        batch : dict
            a batch generated by `ProteinDataset` and `PadCollate`
        lower_limit : int, default 15
            the minimum number of residues to mask
        upper_limit : int, default 100
            the maximum number of residues to mask
        mask_frac : float, optional
            if given, the `lower_limit` and `upper_limit` are ignored and the number of residues to mask is `mask_frac` times the length of the chain
        mask_whole_chains : bool, default False
            if `True`, `upper_limit`, `force_binding_sites` and `lower_limit` are ignored and the whole chain is masked instead
        force_binding_sites_frac : float, default 0.15
            if > 0, in the fraction of cases where a chain from a polymer is sampled, the center of the masked region will be
            forced to be in a binding site

        Returns
        -------
        chain_M : torch.Tensor
            a `(B, L)` shaped binary tensor where 1 denotes the part that needs to be predicted and
            0 is everything else
        """

        super().__init__()
        self.mask_residues = mask_residues
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mask_frac = mask_frac
        self.mask_whole_chains = mask_whole_chains
        self.force_binding_sites_frac = force_binding_sites_frac

    def _get_masked_sequence(
        self,
        batch,
    ):
        """
        Get the mask for the residues that need to be predicted

        Depending on the parameters the residues are selected as follows:
        - if `mask_whole_chains` is `True`, the whole chain is masked
        - if `mask_frac` is given, the number of residues to mask is `mask_frac` times the length of the chain,
        - otherwise, the number of residues to mask is sampled uniformly from the range [`lower_limit`, `upper_limit`].

        If `force_binding_sites_frac` > 0 and `mask_whole_chains` is `False`, in the fraction of cases where a chain
        from a polymer is sampled, the center of the masked region will be forced to be in a binding site.

        Parameters
        ----------
        batch : dict
            a batch generated by `ProteinDataset` and `PadCollate`
        lower_limit : int, default 15
            the minimum number of residues to mask
        upper_limit : int, default 100
            the maximum number of residues to mask
        mask_frac : float, optional
            if given, the `lower_limit` and `upper_limit` are ignored and the number of residues to mask is `mask_frac` times the length of the chain
        mask_whole_chains : bool, default False
            if `True`, `upper_limit`, `force_binding_sites` and `lower_limit` are ignored and the whole chain is masked instead
        force_binding_sites_frac : float, default 0.15
            if > 0, in the fraction of cases where a chain from a polymer is sampled, the center of the masked region will be
            forced to be in a binding site

        Returns
        -------
        chain_M : torch.Tensor
            a `(B, L)` shaped binary tensor where 1 denotes the part that needs to be predicted and
            0 is everything else
        """

        chain_M = torch.zeros(batch["S"].shape)
        interface_lengths = []
        non_masked_interface_lengths = []
        for i, coords in enumerate(batch["X"]):
            chain_index = batch["chain_id"][i]
            chain_bool = batch["chain_encoding_all"][i] == chain_index

            if self.mask_whole_chains:
                chain_M[i, chain_bool] = 1
            else:
                chains = torch.unique(batch["chain_encoding_all"][i])
                chain_start = torch.where(chain_bool)[0][0]
                chain = coords[chain_bool]
                res_i = None
                interface = []
                non_masked_interface = []
                if len(chains) > 1 and self.force_binding_sites_frac > 0:
                    if random.uniform(0, 1) <= self.force_binding_sites_frac:
                        # if torch.cuda.is_available() and coords.device != torch.device('cuda'):
                        #     X_copy = coords.cuda()
                        # else:
                        X_copy = coords

                        i_indices = (chain_bool == 0).nonzero().flatten()
                        j_indices = chain_bool.nonzero().flatten()

                        distances = torch.norm(
                            X_copy[i_indices, 2, :]
                            - X_copy[j_indices, 2, :].unsqueeze(1),
                            dim=-1,
                        ).cpu()
                        # all_distances = torch.norm(X_copy[:, 2, :] - X_copy[:, 2, :].unsqueeze(1), dim=-1)

                        # close_idx = np.where(torch.min(all_distances[:, i_indices][j_indices, :], dim = 1)[0] <  10)[0]
                        close_idx = (
                            np.where(torch.min(distances, dim=1)[0] <= 10)[0]
                            + chain_start.item()
                        )

                        no_mask_idx = np.where(batch["mask"][i][chain_bool])[0]
                        # mask_idx = np.where(batch['mask'][i] == 0)[0]
                        interface = np.intersect1d(close_idx, j_indices)

                        not_end_mask = np.where(
                            ((X_copy[:, 2, :].cpu() == 0).sum(-1) != 3)
                        )[0]
                        interface = np.intersect1d(interface, not_end_mask)

                        non_masked_interface = np.intersect1d(interface, no_mask_idx)
                        interpolate = True
                        if len(non_masked_interface) > 0:
                            res_i = non_masked_interface[
                                random.randint(0, len(non_masked_interface) - 1)
                            ]
                        elif len(interface) > 0 and interpolate:
                            res_i = interface[random.randint(0, len(interface) - 1)]
                        else:
                            res_i = no_mask_idx[random.randint(0, len(no_mask_idx) - 1)]
                if res_i is None:
                    non_zero = torch.where(batch["mask"][i][chain_bool])[0]
                    res_i = non_zero[random.randint(0, len(non_zero) - 1)]
                res_coords = coords[res_i, 2, :]
                neighbor_indices = torch.where(batch["mask"][i][chain_bool])[0]
                if self.mask_frac is not None:
                    assert self.mask_frac > 0 and self.mask_frac < 1
                    k = int(len(neighbor_indices) * self.mask_frac)
                    k = max(k, 10)
                else:
                    up = min(
                        self.upper_limit, int(len(neighbor_indices) * 0.5)
                    )  # do not mask more than half of the sequence
                    low = min(up - 1, self.lower_limit)
                    k = random.choice(range(low, up))
                dist = torch.norm(
                    chain[neighbor_indices, 2, :] - res_coords.unsqueeze(0), dim=-1
                )
                closest_indices = neighbor_indices[
                    torch.topk(dist, k, largest=False)[1]
                ]
                chain_M[i, closest_indices + chain_start] = 1
                # interface_lengths.append(len(interface))
                # non_masked_interface_lengths.append(len(non_masked_interface))
        return chain_M
        # return [chain_M, interface_lengths, non_masked_interface_lengths] # , torch.Tensor(interface), torch.Tensor(non_masked_interface)

    def pad_collate(self, batch):
        # find longest sequence
        out = {}
        max_len = max(map(lambda x: x["S"].shape[0], batch))

        # pad according to max_len
        to_pad = [max_len - b["S"].shape[0] for b in batch]
        for key in batch[0].keys():
            if key in ["chain_id", "chain_dict", "pdb_id"]:
                continue
            out[key] = torch.stack(
                [
                    torch.cat([b[key], torch.zeros((pad, *b[key].shape[1:]))], 0)
                    for b, pad in zip(batch, to_pad)
                ],
                0,
            )
        out["chain_id"] = torch.tensor([b["chain_id"] for b in batch])
        out["masked_res"] = self._get_masked_sequence(out)
        out["chain_dict"] = [b["chain_dict"] for b in batch]
        out["pdb_id"] = [b["pdb_id"] for b in batch]
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
        the path to the folder that will store proteinflow datasets, logs and temporary files
    skip_splitting : bool, default False
        if `True`, skip the split dictionary creation and the file moving steps
    """

    data_path = _download_dataset(tag, local_datasets_folder)
    if not skip_splitting:
        _split_data(data_path)


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

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    _check_mmseqs()
    filter_methods = not not_filter_methods
    remove_redundancies = not not_remove_redundancies
    tmp_folder = os.path.join(local_datasets_folder, "tmp")
    output_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")
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
        load_live=load_live,
    )
    if not skip_splitting:
        _get_split_dictionaries(
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            out_split_dict_folder=out_split_dict_folder,
            min_seq_id=min_seq_id,
        )

        _split_data(output_folder)
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

    Biounits that contain chains similar to those in the `exclude_chains` list (with `exclude_threshold` sequence identity)
    are excluded from the splitting and placed into a separate folder.


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

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors
    """

    if exclude_chains is None:
        excluded_biounits = []
    else:
        excluded_biounits = _get_excluded_files(
            tag,
            local_datasets_folder,
            os.path.join(local_datasets_folder, "tmp"),
            exclude_chains,
            exclude_threshold,
        )

    _check_mmseqs()
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
        _get_split_dictionaries(
            tmp_folder=tmp_folder,
            output_folder=output_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            out_split_dict_folder=out_split_dict_folder,
            min_seq_id=min_seq_id,
        )

    _split_data(output_folder, excluded_biounits)


class ProteinDataset(Dataset):
    """
    Dataset to load proteinflow data

    Saves the model input tensors as pickle files in `features_folder`. When `clustering_dict_path` is provided,
    at each iteration a random bionit from a cluster is sampled.

    If a complex contains multiple chains, they are concatenated. The sequence identity information is preserved in the
    `'chain_encoding_all'` object and in the `'residue_idx'` arrays the chain change is denoted by a +100 jump.

    Returns dictionaries with the following keys and values (all values are `torch` tensors):

    - `'X'`: 3D coordinates of N, C, Ca, O, `(total_L, 4, 3)`,
    - `'S'`: sequence indices (shape `(total_L)`),
    - `'mask'`: residue mask (0 where coordinates are missing, 1 otherwise; with interpolation 0s are replaced with 1s), `(total_L)`,
    - `'mask_original'`: residue mask (0 where coordinates are missing, 1 otherwise; not changed with interpolation), `(total_L)`,
    - `'residue_idx'`: residue indices (from 0 to length of sequence, +100 where chains change), `(total_L)`,
    - `'chain_encoding_all'`: chain indices, `(total_L)`,
    - `'chain_id`': a sampled chain index,
    - `'chain_dict'`: a dictionary of chain ids (keys are chain ids, e.g. `'A'`, values are the indices used in `'chain_id'` and `'chain_encoding_all'` objects)

    You can also choose to include additional features (set in the `node_features_type` parameter):

    - `'sidechain_orientation'`: a unit vector in the direction of the sidechain, `(total_L, 3)`,
    - `'dihedral'`: the dihedral angles, `(total_L, 2)`,
    - `'chemical'`: hydropathy, volume, charge, polarity, acceptor/donor features, `(total_L, 6)`,
    - `'secondary_structure'`: a one-hot encoding of secondary structure ([alpha-helix, beta-sheet, coil]), `(total_L, 3)`,
    - `'sidechain_coords'`: the coordinates of the sidechain atoms (see `proteinflow.sidechain_order()` for the order), `(total_L, 10, 3)`,

    In order to compute additional features, use the `feature_functions` parameter. It should be a dictionary with keys
    corresponding to the feature names and values corresponding to the functions that compute the features. The functions
    should take a chain dictionary and an integer representation of the sequence as input (the dictionary is in `proteinflow` format,
    see the docs for `generate_data` for details) and return a `numpy` array shaped as `(#residues, #features)`.

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
        entry_type="biounit",  # biounit, chain, pair
        classes_to_exclude=None,  # heteromers, homomers, single_chains
        shuffle_clusters=True,
        feature_functions=None,
    ):
        """
        Parameters
        ----------
        dataset_folder : str
            the path to the folder with proteinflow format input files (assumes that files are named {biounit_id}.pickle)
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
        node_features_type : {"zeros", "dihedral", "sidechain_orientation", "chemical", "secondary_structure" or combinations with "+"}
            the type of node features, e.g. `"dihedral"` or `"sidechain_orientation+chemical"`
        debug_file_path : str, optional
            if not `None`, open this single file instead of loading the dataset
        entry_type : {"biounit", "chain", "pair"}
            the type of entries to generate (`"biounit"` for biounit-level complexes, `"chain"` for chain-level, `"pair"`
            for chain-chain pairs (all pairs that are seen in the same biounit and have intersecting coordinate clouds))
        classes_to_exclude : list of str, optional
            a list of classes to exclude from the dataset (select from `"single_chains"`, `"heteromers"`, `"homomers"`)
        shuffle_clusters : bool, default True
            if `True`, a new representative is randomly selected for each cluster at each epoch (if `clustering_dict_path` is given)
        feature_functions : dict, optional
            a dictionary of functions to compute additional features (keys are the names of the features, values are the functions)
        """

        alphabet = ALPHABET
        self.alphabet_dict = defaultdict(lambda: 0)
        for i, letter in enumerate(alphabet):
            self.alphabet_dict[letter] = i
        self.alphabet_dict["X"] = 0
        self.files = defaultdict(lambda: defaultdict(list))  # file path by biounit id
        self.loaded = None
        self.dataset_folder = dataset_folder
        self.features_folder = features_folder
        self.feature_types = []
        if node_features_type is not None:
            self.feature_types = node_features_type.split("+")
        self.entry_type = entry_type
        self.shuffle_clusters = shuffle_clusters
        self.feature_functions = {
            "sidechain_orientation": self._sidechain,
            "dihedral": self._dihedral,
            "chemical": self._chemical,
            "secondary_structure": self._sse,
            "sidechain_coords": self._sidechain_coords,
        }
        self.feature_functions.update(feature_functions or {})
        if classes_to_exclude is not None and not all(
            [
                x in ["single_chains", "heteromers", "homomers"]
                for x in classes_to_exclude
            ]
        ):
            raise ValueError(
                "Invalid class to exclude, choose from 'single_chains', 'heteromers', 'homomers'"
            )

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
        if clustering_dict_path is not None and use_fraction < 1:
            with open(clustering_dict_path, "rb") as f:
                clusters = pickle.load(f)
            keys = sorted(clusters.keys())[: int(len(clusters) * use_fraction)]
            to_process = set()
            for key in keys:
                to_process.update([x[0] for x in clusters[key]])
            to_process = list(to_process)
        if debug:
            to_process = to_process[:1000]
        # output_tuples = [self._process(x, rewrite=rewrite) for x in tqdm(to_process)]
        if self.entry_type == "pair":
            print(
                "Please note that the pair entry type takes longer to process than the other two. The progress bar is not linear because of the varying number of chains per file."
            )
        output_tuples_list = p_map(
            lambda x: self._process(x, rewrite=rewrite, max_length=max_length),
            to_process,
        )
        # save the file names
        for output_tuples in output_tuples_list:
            for id, filename, chain_set in output_tuples:
                for chain in chain_set:
                    self.files[id][chain].append(filename)
        # filter by length
        # seen = set()
        # if max_length is not None:
        #     to_remove = []
        #     for id, chain_dict in self.files.items():
        #         for chain, file_list in chain_dict.items():
        #             for file in file_list:
        #                 if file in seen:
        #                     continue
        #                 seen.add(file)
        #                 with open(file, "rb") as f:
        #                     data = pickle.load(f)
        #                     if len(data["S"]) > max_length:
        #                         to_remove.append(file)
        #     for id in list(self.files.keys()):
        #         chain_dict = self.files[id]
        #         for chain in list(chain_dict.keys()):
        #             file_list = chain_dict[chain]
        #             for file in file_list:
        #                 if file in to_remove:
        #                     self.files[id][chain].remove(file)
        #                     if len(self.files[id][chain]) == 0:
        #                         self.files[id].pop(chain)
        #                     if len(self.files[id]) == 0:
        #                         self.files.pop(id)
        # load the clusters
        if classes_to_exclude is None:
            classes_to_exclude = []
        elif clustering_dict_path is None:
            raise ValueError(
                "classes_to_exclude is not None, but clustering_dict_path is None"
            )
        if clustering_dict_path is not None:
            if entry_type == "pair":
                classes_to_exclude = set(classes_to_exclude)
                classes_to_exclude.add("single_chains")
                classes_to_exclude = list(classes_to_exclude)
            with open(clustering_dict_path, "rb") as f:
                self.clusters = pickle.load(f)  # list of biounit ids by cluster id
                classes = pickle.load(f)
            to_exclude = set()
            for c in classes_to_exclude:
                for key, id_arr in classes.get(c, {}).items():
                    for id, _ in id_arr:
                        to_exclude.add(id)
            for key in list(self.clusters.keys()):
                cluster_list = []
                for x in self.clusters[key]:
                    if x[0] in to_exclude:
                        continue
                    id = x[0].split(".")[0]
                    chain = x[1]
                    if id not in self.files:
                        continue
                    if chain not in self.files[id]:
                        continue
                    if len(self.files[id][chain]) == 0:
                        continue
                    cluster_list.append([id, chain])
                self.clusters[key] = cluster_list
                if len(self.clusters[key]) == 0:
                    self.clusters.pop(key)
            self.data = list(self.clusters.keys())
        else:
            self.clusters = None
            self.data = list(self.files.keys())
        # create a smaller dataset if necessary
        if clustering_dict_path is None and use_fraction < 1:
            self.data = sorted(self.data)[: int(len(self.data) * use_fraction)]
        if load_to_ram:
            print("Loading to RAM...")
            self.loaded = {}
            seen = set()
            for id in self.files:
                for chain, file_list in self.files[id].items():
                    for file in file_list:
                        if file in seen:
                            continue
                        seen.add(file)
                        with open(file, "rb") as f:
                            self.loaded[file] = pickle.load(f)

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

    def _dihedral(self, chain_dict, seq):
        """
        Dihedral angles
        """

        crd = chain_dict["crd_bb"]
        msk = chain_dict["msk"]
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

    def _sidechain(self, chain_dict, seq):
        """
        Sidechain orientation (defined by the 'main atoms' in the `main_atom_dict` dictionary)
        """

        crd_sc = chain_dict["crd_sc"]
        crd_bb = chain_dict["crd_bb"]
        orientation = np.zeros((crd_sc.shape[0], 3))
        for i in range(1, 21):
            if self.main_atom_dict[i] is not None:
                orientation[seq == i] = (
                    crd_sc[seq == i, self.main_atom_dict[i], :] - crd_bb[seq == i, 2, :]
                )
            else:
                S_mask = seq == i
                orientation[S_mask] = np.random.rand(*orientation[S_mask].shape)
        orientation /= np.expand_dims(linalg.norm(orientation, axis=-1), -1) + 1e-7
        return orientation

    def _chemical(self, chain_dict, seq):
        """
        Chemical features (hydropathy, volume, charge, polarity, acceptor/donor)
        """

        features = np.array([_PMAP(x) for x in seq])
        return features

    def _sse(self, chain_dict, seq):
        """
        Secondary structure features
        """

        sse_map = {"c": [0, 0, 1], "b": [0, 1, 0], "a": [1, 0, 0], "": [0, 0, 0]}
        sse = _annotate_sse(chain_dict["crd_bb"])
        sse = np.array([sse_map[x] for x in sse]) * chain_dict["msk"][:, None]
        return sse

    def _sidechain_coords(self, chain_dict, seq):
        """
        Sidechain coordinates
        """

        crd_sc = chain_dict["crd_sc"]
        return crd_sc

    def _process(self, filename, rewrite=False, max_length=None):
        """
        Process a proteinflow file and save it as ProteinMPNN features
        """

        input_file = os.path.join(self.dataset_folder, filename)
        no_extension_name = filename.split(".")[0]
        with open(input_file, "rb") as f:
            data = pickle.load(f)
        chains = sorted(data.keys())
        if self.entry_type == "biounit":
            chain_sets = [chains]
        elif self.entry_type == "chain":
            chain_sets = [[x] for x in chains]
        elif self.entry_type == "pair":
            chain_sets = list(combinations(chains, 2))
        else:
            raise RuntimeError(
                "Unknown entry type, please choose from ['biounit', 'chain', 'pair']"
            )
        output_names = []
        for chains_i, chain_set in enumerate(chain_sets):
            output_file = os.path.join(
                self.features_folder, no_extension_name + f"_{chains_i}.pickle"
            )
            pass_set = False
            add_name = True
            if os.path.exists(output_file) and not rewrite:
                pass_set = True
                if max_length is not None:
                    if sum([len(data[x]["seq"]) for x in chain_set]) > max_length:
                        add_name = False
            else:
                X = []
                S = []
                mask = []
                mask_original = []
                chain_encoding_all = []
                residue_idx = []
                node_features = defaultdict(lambda: [])
                last_idx = 0
                chain_dict = {}

                if max_length is not None:
                    if sum([len(data[x]["seq"]) for x in chain_set]) > max_length:
                        pass_set = True
                        add_name = False

                if self.entry_type == "pair":
                    # intersect = []
                    X1 = data[chain_set[0]]["crd_bb"][
                        data[chain_set[0]]["msk"].astype(bool)
                    ]
                    X2 = data[chain_set[1]]["crd_bb"][
                        data[chain_set[1]]["msk"].astype(bool)
                    ]
                    intersect_dim_X1 = []
                    intersect_dim_X2 = []
                    intersect_X1 = np.zeros(len(X1))
                    intersect_X2 = np.zeros(len(X2))
                    margin = 30
                    cutoff = 10
                    for dim in range(3):
                        min_dim_1 = X1[:, 2, dim].min()
                        max_dim_1 = X1[:, 2, dim].max()
                        min_dim_2 = X2[:, 2, dim].min()
                        max_dim_2 = X2[:, 2, dim].max()
                        intersect_dim_X1.append(
                            np.where(
                                np.logical_and(
                                    X1[:, 2, dim] >= min_dim_2 - margin,
                                    X1[:, 2, dim] <= max_dim_2 + margin,
                                )
                            )[0]
                        )
                        intersect_dim_X2.append(
                            np.where(
                                np.logical_and(
                                    X2[:, 2, dim] >= min_dim_1 - margin,
                                    X2[:, 2, dim] <= max_dim_1 + margin,
                                )
                            )[0]
                        )

                        # if min_dim_1 - 4 <= max_dim_2 and max_dim_1 >= min_dim_2 - 4:
                        #     intersect.append(True)
                        # else:
                        #     intersect.append(False)
                        #     break
                    intersect_X1 = np.intersect1d(
                        np.intersect1d(intersect_dim_X1[0], intersect_dim_X1[1]),
                        intersect_dim_X1[2],
                    )
                    intersect_X2 = np.intersect1d(
                        np.intersect1d(intersect_dim_X2[0], intersect_dim_X2[1]),
                        intersect_dim_X2[2],
                    )

                    not_end_mask1 = np.where(((X1[:, 2, :] == 0).sum(-1) != 3))[0]
                    not_end_mask2 = np.where(((X2[:, 2, :] == 0).sum(-1) != 3))[0]

                    intersect_X1 = np.intersect1d(intersect_X1, not_end_mask1)
                    intersect_X2 = np.intersect1d(intersect_X2, not_end_mask2)

                    # distances = torch.norm(X1[intersect_X1, 2, :] - X2[intersect_X2, 2, :](1), dim=-1)
                    diff = X1[intersect_X1, 2, np.newaxis, :] - X2[intersect_X2, 2, :]
                    distances = np.sqrt(np.sum(diff**2, axis=2))

                    intersect_X1 = torch.LongTensor(intersect_X1)
                    intersect_X2 = torch.LongTensor(intersect_X2)
                    if np.sum(distances < cutoff) < 3:
                        # if not all(intersect):
                        pass_set = True
                        add_name = False
            if add_name:
                output_names.append(
                    (os.path.basename(no_extension_name), output_file, chain_set)
                )
            if pass_set:
                continue

            for chain_i, chain in enumerate(chain_set):
                seq = torch.tensor([self.alphabet_dict[x] for x in data[chain]["seq"]])
                S.append(seq)
                mask_original.append(deepcopy(data[chain]["msk"]))
                if self.interpolate != "none":
                    data[chain]["crd_bb"], data[chain]["msk"] = self._interpolate(
                        data[chain]["crd_bb"], data[chain]["msk"]
                    )
                X.append(data[chain]["crd_bb"])
                mask.append(data[chain]["msk"])
                residue_idx.append(torch.arange(len(data[chain]["seq"])) + last_idx)
                last_idx = residue_idx[-1][-1] + 100
                chain_encoding_all.append(torch.ones(len(data[chain]["seq"])) * chain_i)
                chain_dict[chain] = chain_i
                for name in self.feature_types:
                    if name not in self.feature_functions:
                        continue
                    func = self.feature_functions[name]
                    node_features[name].append(func(data[chain], seq))

            out = {}
            out["X"] = torch.from_numpy(np.concatenate(X, 0))
            out["S"] = torch.cat(S)
            out["mask"] = torch.from_numpy(np.concatenate(mask))
            out["mask_original"] = torch.from_numpy(np.concatenate(mask_original))
            out["chain_encoding_all"] = torch.cat(chain_encoding_all)
            out["residue_idx"] = torch.cat(residue_idx)
            out["chain_dict"] = chain_dict
            out["pdb_id"] = no_extension_name.split("-")[0]
            for key, value_list in node_features.items():
                out[key] = torch.from_numpy(np.concatenate(value_list))
            with open(output_file, "wb") as f:
                pickle.dump(out, f)
        return output_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chain_id = None
        if self.clusters is None:
            id = self.data[idx]  # data is already filtered by length
            chain_id = random.choice(list(self.files[id].keys()))
        else:
            cluster = self.data[idx]
            id = None
            chain_n = -1
            # print(f'{self.clusters[cluster]=}')
            while (
                id is None or len(self.files[id][chain_id]) == 0
            ):  # some IDs can be filtered out by length
                if self.shuffle_clusters:
                    chain_n = random.randint(0, len(self.clusters[cluster]) - 1)
                else:
                    chain_n += 1
                id, chain_id = self.clusters[cluster][
                    chain_n
                ]  # get id and chain from cluster
                # print(f'{id=}, {len(self.files[id][chain_id])=}')
        file = random.choice(self.files[id][chain_id])
        if self.loaded is None:
            with open(file, "rb") as f:
                try:
                    data = pickle.load(f)
                except EOFError:
                    print("EOFError", file)
                    raise
        else:
            data = deepcopy(self.loaded[file])
        data["chain_id"] = data["chain_dict"][chain_id]
        return data


class ProteinLoader(DataLoader):
    """
    A subclass of `torch.data.utils.DataLoader` tuned for the `proteinflow` dataset

    Creates and iterates over an instance of `ProteinDataset`, omitting the `'chain_dict'` keys.
    See the `ProteinDataset` docs for more information.

    If batch size is larger than one, all objects are padded with zeros at the ends to reach the length of the
    longest protein in the batch.

    If `mask_residues` is `True`, an additional `'masked_res'` key is added to the output. The value is a binary
    tensor shaped `(B, L)` where 1 denotes the part that needs to be predicted and 0 is everything else. The tensors are generated
    according to the following rules:
    - if `mask_whole_chains` is `True`, the whole chain is masked
    - if `mask_frac` is given, the number of residues to mask is `mask_frac` times the length of the chain,
    - otherwise, the number of residues to mask is sampled uniformly from the range [`lower_limit`, `upper_limit`].

    If `force_binding_sites_frac` > 0 and `mask_whole_chains` is `False`, in the fraction of cases where a chain
    from a polymer is sampled, the center of the masked region will be forced to be in a binding site.
    """

    def __init__(
        self,
        dataset,
        lower_limit=15,
        upper_limit=100,
        mask_residues=True,
        mask_whole_chains=False,
        mask_frac=None,
        collate_func=_PadCollate,
        force_binding_sites_frac=0,
        shuffle_batches=True,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dataset : ProteinDataset
            a ProteinDataset instance
        lower_limit : int, default 15
            the minimum number of residues to mask
        upper_limit : int, default 100
            the maximum number of residues to mask
        mask_frac : float, optional
            if given, the `lower_limit` and `upper_limit` are ignored and the number of residues to mask is `mask_frac` times the length of the chain
        mask_whole_chains : bool, default False
            if `True`, `upper_limit`, `force_binding_sites` and `lower_limit` are ignored and the whole chain is masked instead
        force_binding_sites_frac : float, default 0
            if > 0, in the fraction of cases where a chain from a polymer is sampled, the center of the masked region will be
            forced to be in a binding site
        shuffle_clusters : bool, default True
            if `True`, a new representative is randomly selected for each cluster at each epoch (if `clustering_dict_path` is given)
        shuffle_batches : bool, default True
            if `True`, the batches are shuffled at each epoch
        collate_func : callable, optional
            a function that takes a list of samples and returns a batch and inherits from _PadCollate
        """

        super().__init__(
            dataset,
            collate_fn=collate_func(
                mask_residues=mask_residues,
                mask_whole_chains=mask_whole_chains,
                mask_frac=mask_frac,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                force_binding_sites_frac=force_binding_sites_frac,
            ),
            shuffle=shuffle_batches,
            *args,
            **kwargs,
        )

    @staticmethod
    def from_args(
        dataset_folder,
        features_folder="./data/tmp/",
        clustering_dict_path=None,
        max_length=None,
        rewrite=False,
        use_fraction=1,
        load_to_ram=False,
        debug=False,
        interpolate="none",
        node_features_type=None,
        entry_type="biounit",  # biounit, chain, pair
        classes_to_exclude=None,
        lower_limit=15,
        upper_limit=100,
        mask_residues=True,
        mask_whole_chains=False,
        mask_frac=None,
        force_binding_sites_frac=0,
        shuffle_clusters=True,
        shuffle_batches=True,
        *args,
        **kwargs,
    ) -> None:
        """
        Creates a `ProteinLoader` instance with a `ProteinDataset` from the given arguments

        Parameters
        ----------
        dataset_folder : str
            the path to the folder with proteinflow format input files (assumes that files are named {biounit_id}.pickle)
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
        node_features_type : {"dihedral", "sidechain_orientation", "chemical", "secondary_structure", "sidechain_coords", or combinations with "+"}, optional
            the type of node features, e.g. `"dihedral"` or `"sidechain_orientation+chemical"`
        entry_type : {"biounit", "chain", "pair"}
            the type of entries to generate (`"biounit"` for biounit-level, `"chain"` for chain-level, `"pair"` for chain-chain pairs)
        classes_to_exclude : list of str, optional
            a list of classes to exclude from the dataset (select from `"single_chains"`, `"heteromers"`, `"homomers"`)
        lower_limit : int, default 15
            the minimum number of residues to mask
        upper_limit : int, default 100
            the maximum number of residues to mask
        mask_frac : float, optional
            if given, the `lower_limit` and `upper_limit` are ignored and the number of residues to mask is `mask_frac` times the length of the chain
        mask_whole_chains : bool, default False
            if `True`, `upper_limit`, `force_binding_sites` and `lower_limit` are ignored and the whole chain is masked instead
        force_binding_sites_frac : float, default 0
            if > 0, in the fraction of cases where a chain from a polymer is sampled, the center of the masked region will be
            forced to be in a binding site
        shuffle_clusters : bool, default True
            if `True`, a new representative is randomly selected for each cluster at each epoch (if `clustering_dict_path` is given)
        shuffle_batches : bool, default True
            if `True`, the batches are shuffled at each epoch
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
            entry_type=entry_type,
            classes_to_exclude=classes_to_exclude,
            shuffle_clusters=shuffle_clusters,
        )
        return ProteinLoader(
            dataset=dataset,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            mask_residues=mask_residues,
            mask_whole_chains=mask_whole_chains,
            mask_frac=mask_frac,
            force_binding_sites_frac=force_binding_sites_frac,
            shuffle_batches=shuffle_batches,
            *args,
            **kwargs,
        )


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
