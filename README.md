# BestProt
This is a python library for handling the BestProt data processing pipeline.

## Installation
Recommended: create a new `conda` environment and install `bestprot` and `mmseqs`. Note that the python version has to be between 3.8 and 3.10.
```
git clone https://gitlab.com/adaptyvbio/ml-4/-/tree/library
cd ml-4
conda create --name bestprot -y python=3.9
conda activate bestprot
conda install -c conda-forge -c bioconda mmseqs2
python -m pip install -e .
```

## Usage
### Downloading pre-computed datasets
We have run the pipeline and saved the results at an AWS S3 server. You can download the resulting dataset with `bestprot`.
```
download_bestprot --tag 20221110 
```

### Running the pipeline
You can also run `bestprot` with your own parameters.
```
generate_bestprot --tag new --resolution_thr 5 --pdb_snapshot 20190101 --not_filter_methods
```
See the docs (or `generate_bestrot --help`) for the full list of parameters.

### Splitting


### Using the data
The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are the following:
- `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
- `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (check `bestprot.sidechain_order()` for the order of atoms),
- `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and
    zeros to missing values,
- `'seq'`: a string of length `L` with residue types.

Once you've downloaded or generated your data, you can use our `ProteinDataset` or `ProteinLoader` classes 
for convenient processing. 

## Data

|Date    |Location (S3)|Size|Min res|Min len|Max len|ID threshold|Split (train/val/test)|Missing thr (ends/middle)|
|--------|--------|----|-------|-------|-------|------------|-----|-----------|
|10.11.22|[data](s3://ml4-main-storage/bestprot_20221110/) [split]("s3://ml4-main-storage/bestprot_20221110_splits_dict/")|24G|3.5|30|10000|0.9|90/5/5|0.3/0.1



