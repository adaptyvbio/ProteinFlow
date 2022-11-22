# ml-4
Equivariant NN for generative sequence and structure antibody co-design

## Generating data
In order to download and process PDB files, first install the neccessary requirements;
```
git clone https://gitlab.com/adaptyvbio/ml-4
cd ml-4
git submodule init
git submodule update
conda create -y --name data python=3.9
conda activate data
python -m pip install -e rcsbsearch/ 
python -m pip install -r requirements.txt
```

Then you can run `run_pdb_processing.py`.
```
conda activate data
python run_pdb_processing.py --min_length 30 --resolution_thr 3.5
```

See `python run_pdb_processing.py --help` for a full list of parameters and a description of the file format.

This will generate pickled files for the data (one per biounit), as well as clustering dictionaries. The clustering dictionaries 
are called `train.pickle`, `valid.pickle` and `test.pickle`. Read them like this.
```
with open("train.pickle", "rb") as f:
    cluster_dict = pickle.load(f)
    class_dict = pickle.load(f)
```

The `cluster_dict` will contain information about the MMSeqs clustering and the `class_dict` about the distribution over single chains, homomers and heteromers for each subset.

To download the dataset from S3 and partition into training, validation and test folders according to the dictionaries, run this command.
```
python split_dataset.py --dataset_path s3://path/to/dataset --dict_path s3://path/to/dicts/ --local_folder /local/path
```

Alternatively, if you already have the data on your machine, just use the same command but with local paths (the data will be moved from `dataset_path` to `local_folder` and rearranged into training/test/validation).

## Submodules

To add a repository from github as a submodule, follow this.
1. `git clone [github url]`,
2. `cd [repo name]`,
3. `git remote rename origin upstream`,
4. create a new project at https://gitlab.com/adaptyvbio (be sure to uncheck the README option),
5. `git remote add origin [gitlab url]`,
6. `git push -u origin --all`,
7. `git push -u origin --tags`,
8. go to `ml-4` on your machine,
9. `git submodule add [gitlab url]`.


## Data

|Date    |Location (S3)|Size|Min res|Min len|Max len|ID threshold|Split (train/val/test)|Missing thr (ends/middle)|
|--------|--------|----|-------|-------|-------|------------|-----|-----------|
|10.11.22|[data](s3://ml4-main-storage/bestprot_20221110/) [split]("s3://ml4-main-storage/bestprot_20221110_splits_dict/")|24G|3.5|30|10000|0.9|90/5/5|0.3/0.1



