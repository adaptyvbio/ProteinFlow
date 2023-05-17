FROM python:3.10-buster
RUN python -m pip install --upgrade pip setuptools wheel
RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz; tar xvfz mmseqs-linux-avx2.tar.gz;
RUN python -m pip install prody==2.4.0
RUN python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
RUN python -m pip install proteinflow
RUN echo "export PATH=$(pwd)/mmseqs/bin/:$PATH" >> ~/.bashrc
