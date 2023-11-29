FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y wget libxml2 cuda-minimal-build-11-7 libcusparse-dev-11-7 libcublas-dev-11-7 libcusolver-dev-11-7 git
RUN wget -P /tmp \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH /opt/conda/bin:$PATH
RUN export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

RUN mamba install -y python=3.9
RUN mamba install -y -c conda-forge openmm==7.7.0 pdbfixer
RUN mamba install -y -c bioconda anarci
RUN python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN python -m pip install "fair-esm[esmfold]"
RUN python -m pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN python -m pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz; tar xvfz mmseqs-linux-avx2.tar.gz;
RUN wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz; tar xvzf foldseek-linux-avx2.tar.gz;
RUN python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
RUN python -m pip install proteinflow[processing]
ENV PATH /foldseek/bin/:/mmseqs/bin/:$PATH
