FROM python:3.9-slim-buster
RUN apt-get update
RUN apt-get install -y git-all wget
RUN wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz; tar xvfz mmseqs-linux-avx2.tar.gz
RUN python -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"
RUN python -m pip install proteinflow
CMD ["export", "PATH=$(pwd)/mmseqs/bin/:$PATH"]