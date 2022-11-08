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


