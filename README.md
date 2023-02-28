# ProteinFlow

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/pypi/v/proteinflow)](https://pypi.org/project/proteinflow/)
[![Docker Image Version (latest semver)](https://img.shields.io/docker/v/adaptyvbio/proteinflow?label=docker)](https://hub.docker.com/r/adaptyvbio/proteinflow/tags)
![Generic badge](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

A data processing pipeline for all your protein design needs.

[Read the documentation.](https://adaptyvbio.github.io/ProteinFlow/)

![overview](https://raw.githubusercontent.com/adaptyvbio/ProteinFlow/main/media/fig_pipeline.png)

## Installation
Recommended: create a new `conda` environment and install `proteinflow` and `mmseqs`. Note that the python version has to be between 3.8 and 3.10. 
```bash
conda create --name proteinflow -y python=3.9
conda activate proteinflow
conda install -y -c conda-forge -c bioconda mmseqs2
python -m pip install proteinflow
```
In addition, `proteinflow` depends on the `rcsbsearch` package and the latest release [v0.2.3](https://github.com/sbliven/rcsbsearch/releases/tag/v0.2.3) is currently not functioning . Follow the recommended fix:
```bash
python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
```

Note that you do not need to install `mmseqs` or `rcsbsearch` if you are not planning to generate a new dataset.

Finally, you can use our [docker image](https://hub.docker.com/r/adaptyvbio/proteinflow/tags) as an alternative.

## Usage
### Downloading pre-computed datasets (stable)
Already precomputed datasets with consensus set of parameters and can be accessed and downloaded using the `proteinflow`. package. Check the output of `proteinflow check_tags` for a list of available tags.
```bash
proteinflow download --tag 20221110 
```

### Running the pipeline
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

### Splitting
By default, both `proteinflow generate` and `proteinflow download` will also split your data into training, test and validation according to MMseqs2 clustering and homomer/heteromer/single chain proportions. However, you can skip this step with a `--skip_splitting` flag and then perform it separately with the `proteinflow split` command.

The following command will perform the splitting with a 10% validation set, a 5% test set and a 50% threshold for sequence identity clusters.
```bash
proteinflow split --tag new --valid_split 0.1 --test_split 0.5 --min_seq_id 0.5
```

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
- has batch size 8.

```python
from proteinflow import ProteinLoader
train_loader = ProteinLoader(
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

|Tag    |Date    |Snapshot|Size|Min res|Min len|Max len|MMseqs thr|Split (train/val/test)|Missing thr (ends/middle)|Note|
|-------|--------|--------|----|-------|-------|-------|----------|----------------------|-------------------------|----|
|paper|10.11.22|20220103|24G|3.5|30|10'000|0.3|90/5/5|0.3/0.1|first release, no mmCIF files|
|20230102_stable|27.02.23|20230102|28G|3.5|30|10'000|0.3|90/5/5|0.3/0.1| v1.1.1|

## License
The `proteinflow` package and data are released and distributed under the BSD 3-Clause License


## Contributions
This is an open source project supported by [Adaptyv Bio](https://www.adaptyvbio.com/). Contributions, suggestions and bug-fixes are welcomed.

