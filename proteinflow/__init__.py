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
- If you are planning to generate new datasets and installed `proteinflow` with `pip` (or with `conda` on Mac OS with an M1 processor), you will need to additionally install [`mmseqs`](https://github.com/soedinglab/MMseqs2).
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

SAbDab sequences clustering is done across all 6 Complementary Determining Regions (CDRs) - H1, H2, H3, L1, L2, L3, based on the [Chothia numbering](https://pubmed.ncbi.nlm.nih.gov/9367782/) implemented by SabDab. CDRs from nanobodies and other synthetic constructs are clustered together with other heavy chain CDRs. The resulting CDR clusters are split into training, test and validation in a way that ensures that every PDB file only appears in one subset.

Individual output pickle files represent heavy chain - light chain - antigen complexes (created from SAbDab annotation, sometimes more than one per PDB entry). Each of the elements (heavy chain, light chain, antigen) can be missing in specific entries and there can be multiple antigen chains. In order to filter for at least one antigen chain, use the `--require_antigen` option.

For instance, let's generate a dataset with the following description:
- SabDab version: latest (up-to-date),
- resolution threshold: 5 angstrom,
- structure methods accepted: all (x-ray christolography, NRM, Cryo-EM),
- sequence identity threshold for clustering (CDRs): 40%,
- size of validation subset: 10%.

```bash
proteinflow generate --sabdab --tag new --resolution_thr 5 --not_filter_methods --min_seq_id 0.4 --valid_split 0.1
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
- `'cdr'`: a `numpy` array of shape `(L,)` where CDR residues are marked with the corresponding type (`'H1'`, `'L1'`, ...)
and non-CDR residues are marked with `'-'`.

Note that the sequence information in the PDB files is aligned to the FASTA sequences to identify the missing residues.

Once your data is ready, you can open the files with `pickle` directly.

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
crd_bb = batch["X"] #(B, L, 4, 3)
seq = batch["S"] #(B, L)
sse = batch["secondary_structure"] #(B, L, 3)
to_predict = batch["masked_res"] #(B, L), 1 where the residues should be masked, 0 otherwise
...
```
See more details on available parameters and the data format in the [docs](https://adaptyvbio.github.io/ProteinFlow/) + [this repository](https://github.com/adaptyvbio/ProteinFlow-models) for a use case.

## ProteinFlow Stable Releases
You can download them with `proteinflow download --tag {tag}` in the command line or browse in the [interface](https://proteinflow-datasets.s3.eu-west-1.amazonaws.com/index.html).

|Tag    |Date    |Snapshot|Size|Min res|Min len|Max len|Max chains|MMseqs thr|Split (train/val/test)|Missing thr (ends/middle)|Source|Remove redundancies|Note|
|-------|--------|--------|----|-------|-------|-------|----------|----------|----------------------|-------------------------|---|---|----------------|
|paper|10.11.22|20220103|24G|3.5|30|10'000|-|0.3|90/5/5|0.3/0.1|PDB|yes|first release, no mmCIF files|
|20230102_stable|27.02.23|20230102|28G|3.5|30|10'000|-|0.3|90/5/5|0.3/0.1|PDB|yes|v1.1.1|
|20230623_sabdab|26.06.23|live 26.06.23|1.4G|3.5|30|10'000|-|0.3|96/3/1|0.5/0.2|SAbDab|no|v1.4.1 (requires >= v1.4.0)|
|20230102_v200|19.07.23|20230102|33G|3.5|30|10'000|10|0.3|90/5/5|0.3/0.1|PDB|no|v2.0.0|

"""
__pdoc__ = {
    "data.utils": False,
    "download.boto": False,
    "constants": False,
    "split": False,
    "cli": False,
    "ligand": False,
}
__docformat__ = "numpy"

import os
import random
import shutil
import string
import tempfile
import warnings

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config

from proteinflow.constants import SIDECHAIN_ORDER
from proteinflow.data.torch import ProteinDataset, ProteinLoader
from proteinflow.download import _download_dataset, _get_chain_pdb_ids
from proteinflow.download.boto import _s3list
from proteinflow.processing import run_processing
from proteinflow.split import (
    _check_mmseqs,
    _exclude_files_with_no_ligand,
    _get_excluded_files,
    _get_split_dictionaries,
    _split_data,
)


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
    tag=None,
    pdb_id_list_path=None,
    local_datasets_folder="./data",
    min_length=30,
    max_length=10000,
    resolution_thr=3.5,
    missing_ends_thr=0.3,
    missing_middle_thr=0.1,
    not_filter_methods=False,
    not_remove_redundancies=False,
    skip_splitting=False,
    redundancy_thr=0.9,
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
    exclude_chains_file=None,
    exclude_threshold=0.7,
    exclude_clusters=False,
    exclude_based_on_cdr=None,
    load_ligands=False,
    exclude_chains_without_ligands=False,
    tanimoto_clustering=False,
    foldseek=False,
    require_ligand=False,
    random_seed=42,
    max_chains=10,
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
    tag : str, optional
        the name of the dataset to load
    pdb_id_list_path : str, optional
        if provided, get pdb_ids from text file where each line contains one chain id (format pdb_id-num example: 1XYZ-1)
        pdb_ids can also be passed, an automatic retrieval of chain id s will be performed in that case
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
    redundancy_thr : float, default 0.9
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
    exclude_chains_file : str, optional
        path to a file containing the sequences to exclude, one sequence per line
    exclude_threshold : float in [0, 1], default 0.7
        the sequence similarity threshold for excluding chains
    exclude_clusters : bool, default False
        if `True`, exclude clusters that contain chains similar to chains in the `exclude_chains` list
    exclude_based_on_cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
        if given and `exclude_clusters` is `True` + the dataset is SAbDab, exclude files based on only the given CDR clusters
    load_ligands : bool, default False
        if `True`, load ligands from the PDB files
    exclude_chains_without_ligands : bool, default False
        if `True`, exclude biounits that don't contain ligands
    tanimoto_clustering : bool, default False
        if `True`, cluster the biounits based on ligand Tanimoto similarity
    foldseek : bool, default False
        if `True`, cluster the biounits based on structure similarity
    require_ligand : bool, default False
        if `True`, only use biounits that contain a ligand
    random_seed : int, default 42
        the random seed to use for splitting
    max_chains : int, default 10
        the maximum number of chains per biounit

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    tmp_id = "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(5)
    )
    if tag is None:
        if pdb_id_list_path is None:
            raise RuntimeError(
                "Please input a data source: valid tag or a pdb ids list"
            )
        else:
            tag = pdb_id_list_path.split("/")[-1].split(".")[0]
            tmp_folder = os.path.join(tempfile.gettempdir(), tag + tmp_id)
            os.makedirs(tmp_folder)
            with open(pdb_id_list_path) as file:
                # Read lines from the file
                example_pdb_id = file.readline()
            if "-" not in example_pdb_id:
                pdb_id_list_path = _get_chain_pdb_ids(pdb_id_list_path, tmp_folder)
    else:
        tmp_folder = os.path.join(tempfile.gettempdir(), tag + tmp_id)
        os.makedirs(tmp_folder)
    filter_methods = not not_filter_methods
    remove_redundancies = not not_remove_redundancies

    output_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")

    if force and os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    log_dict = run_processing(
        tmp_folder=tmp_folder,
        output_folder=output_folder,
        min_length=min_length,
        max_length=max_length,
        resolution_thr=resolution_thr,
        missing_ends_thr=missing_ends_thr,
        missing_middle_thr=missing_middle_thr,
        filter_methods=filter_methods,
        remove_redundancies=remove_redundancies,
        redundancy_thr=redundancy_thr,
        n=n,
        force=force,
        tag=tag,
        pdb_snapshot=pdb_snapshot,
        load_live=load_live,
        sabdab=sabdab,
        sabdab_data_path=sabdab_data_path,
        require_antigen=require_antigen,
        max_chains=max_chains,
        pdb_id_list_path=pdb_id_list_path,
        load_ligands=load_ligands,
        require_ligand=require_ligand,
    )

    if not skip_splitting:
        if tanimoto_clustering and not load_ligands:
            print(
                "Can not use Tanimoto Clustering without load_ligands=False. Setting tanimoto_clustering to False"
            )
            tanimoto_clustering = False
        split_data(
            tag=tag,
            local_datasets_folder=local_datasets_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            ignore_existing=True,
            min_seq_id=min_seq_id,
            exclude_chains=exclude_chains,
            exclude_chains_file=exclude_chains_file,
            exclude_threshold=exclude_threshold,
            exclude_clusters=exclude_clusters,
            exclude_based_on_cdr=exclude_based_on_cdr,
            random_seed=random_seed,
            exclude_chains_without_ligands=exclude_chains_without_ligands,
            tanimoto_clustering=tanimoto_clustering,
            foldseek=foldseek,
        )
    shutil.rmtree(tmp_folder)
    return log_dict


def split_data(
    tag,
    local_datasets_folder="./data",
    split_tolerance=0.2,
    test_split=0.05,
    valid_split=0.05,
    ignore_existing=False,
    min_seq_id=0.3,
    exclude_chains=None,
    exclude_chains_file=None,
    exclude_threshold=0.7,
    exclude_clusters=False,
    exclude_based_on_cdr=None,
    random_seed=42,
    exclude_chains_without_ligands=False,
    tanimoto_clustering=False,
    foldseek=False,
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
    exclude_chains_file : str, optional
        path to a file containing the sequences to exclude, one sequence per line
    exclude_threshold : float in [0, 1], default 0.7
        the sequence similarity threshold for excluding chains
    exclude_clusters : bool, default False
        if `True`, exclude clusters that contain chains similar to chains in the `exclude_chains` list
    exclude_based_on_cdr : {"H1", "H2", "H3", "L1", "L2", "L3"}, optional
        if given and `exclude_clusters` is `True` + the dataset is SAbDab, exclude files based on only the given CDR clusters
    random_seed : int, default 42
        random seed for reproducibility (set to `None` to use a random seed)
    exclude_chains_without_ligands : bool, default False
        if `True`, exclude biounits that don't contain ligands
    tanimoto_clustering: bool, default False
        cluster chains based on the tanimoto similarity of their ligands
    foldseek: bool, default False
        if `True`, use FoldSeek to cluster chains based on their structure similarity

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    temp_folder = os.path.join(tempfile.gettempdir(), "proteinflow")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    if exclude_chains_file is not None or exclude_chains is not None:
        excluded_biounits = _get_excluded_files(
            tag,
            local_datasets_folder,
            temp_folder,
            exclude_chains,
            exclude_chains_file,
            exclude_threshold,
        )
    else:
        excluded_biounits = []
    if exclude_chains_without_ligands:
        excluded_biounits += _exclude_files_with_no_ligand(
            tag,
            local_datasets_folder,
        )

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
        np.random.seed(random_seed)
        _get_split_dictionaries(
            tmp_folder=temp_folder,
            output_folder=output_folder,
            split_tolerance=split_tolerance,
            test_split=test_split,
            valid_split=valid_split,
            out_split_dict_folder=out_split_dict_folder,
            min_seq_id=min_seq_id,
            tanimoto_clustering=tanimoto_clustering,
            foldseek=foldseek,
        )
    shutil.rmtree(temp_folder)

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
    return [
        x.key.strip("/") for x in folders if not x.key.strip("/").startswith("test")
    ]
