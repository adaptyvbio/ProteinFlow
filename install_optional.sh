conda install -y -c conda-forge openmm==7.7.0 pdbfixer
conda install -y -c bioconda anarci

python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# choose the CUDA version that matches your system (check with `nvcc --version` and get the right command at https://pytorch.org/get-started/previous-versions/)

python -m pip install "fair-esm[esmfold]"
python -m pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
python -m pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

python -m pip install ablang igfold immunebuilder

python -m pip install -e .
python -m pip install ipykernel