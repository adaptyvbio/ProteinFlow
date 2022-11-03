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

See `python run_pdb_processing.py --help` for a full list of parameters.


## Add your files

```
cd existing_repo
git remote add origin https://gitlab.com/adaptyvbio/ml-4.git
git branch -M main
git push -uf origin main
```

## data

|Name|Dataset|Origin|Size|File|Description|
|----|-----|-----|----|-----|------|


