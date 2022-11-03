# ml-4
Equivariant NN for generative sequence and structure antibody co-design

## Generating data
In order to download and process PDB files, first install the neccessary requirements;
```
cd ml-4
conda create --name data python=3.9
conda activate data
python -m pip install -r requirements.txt
git clone https://github.com/sbliven/rcsbsearch
cd rcsbsearch/
git pull origin pull/6/head
python -m pip install . 
cd ..
```

Then you can run `run_pdb_processing.py`.
```
conda activate data
cd utils
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


