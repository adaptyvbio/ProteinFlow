# ProteinFlow

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This is a python library for handling the proteinflow data processing pipeline.

[Read the documentation.](https://adaptyvbio.github.io/ProteinFlow/)

## Installation
Recommended: create a new `conda` environment and install `proteinflow` and `mmseqs`. Note that the python version has to be between 3.8 and 3.10. 
```
conda create --name proteinflow -y python=3.9
conda activate proteinflow
conda install -y -c conda-forge -c bioconda mmseqs2
python -m pip install proteinflow
aws configure
```
In addition, `proteinflow` depends on the `rcsbsearch` package and the latest release is currently failing. Follow the recommended fix:
```
python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
```

## Usage
### Downloading pre-computed datasets
We have run the pipeline and saved the results at an AWS S3 server. You can download the resulting dataset with `proteinflow`. Check the output of `proteinflow check_tags` for a list of available tags.
```
proteinflow download --tag 20221110 
```

### Running the pipeline
You can also run `proteinflow` with your own parameters. Check the output of `proteinflow check_snapshots` for a list of available snapshots.
```
proteinflow generate --tag new --resolution_thr 5 --pdb_snapshot 20190101 --not_filter_methods
```
See the docs (or `proteinflow generate --help`) for the full list of parameters.

The reasons for filtering files out are logged in text files (at `data/logs` by default). To get a summary, run `proteinflow get_summary {log_path}`.

### Splitting
By default, both `proteinflow generate` and `proteinflow download` will also split your data into training, test and validation according to MMseqs2 clustering and homomer/heteromer/single chain proportions. However, you can skip this step with a `--skip_splitting` flag and then perform it separately with the `proteinflow split` command.
```
proteinflow split --tag new --valid_split 0.1 --test_split 0
```

### Using the data
The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are the following:
- `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
- `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (check `proteinflow.sidechain_order()` for the order of atoms),
- `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
    zeros to missing values,
- `'seq'`: a string of length `L` with residue types.

Once your data is ready, you can use our `ProteinDataset` or `ProteinLoader` classes 
for convenient processing. 
```python
from proteinflow import ProteinLoader
train_loader = ProteinLoader("./data/proteinflow_new/training", batch_size=8)
for batch in train_loader:
    ...
```

## Data

|Date    |Location (S3)|Size|Min res|Min len|Max len|ID threshold|Split (train/val/test)|Missing thr (ends/middle)|
|--------|--------|----|-------|-------|-------|------------|-----|-----------|
|10.11.22|[data](s3://ml4-main-storage/proteinflow_20221110/) [split]("s3://ml4-main-storage/proteinflow_20221110_splits_dict/")|24G|3.5|30|10000|0.9|90/5/5|0.3/0.1



