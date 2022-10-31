from re import L
from utils.parse_pdb import align_pdb, open_pdb, PDBError
import os
import boto3
import pickle

TMP_FOLDER = "data/tmp_pdb"
OUTPUT_FOLDER = "data/pdb"
PDB_PREFIX = "20220103/pub/pdb/data/biounit/PDB/all/"

i = 0
while os.path.exists(f"./log_{i}.txt"):
    i += 1
LOG_FILE = f"./log_{i}.txt"

if not os.path.exists(TMP_FOLDER):
    os.mkdir(TMP_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

s3 = boto3.resource('s3')
bucket = s3.Bucket("pdbsnapshots")
all_pdbs = bucket.objects.filter(Prefix=PDB_PREFIX)[:10]


def log_exception(exception, log_file, pdb_id):
    if isinstance(exception, PDBError):
        with open(log_file, "r") as f:
            f.write(f'<<< {str(exception)}: {pdb_id}')
    else:
        with open(log_file, "r") as f:
            f.write(f'<<< Unknown: {pdb_id}')
            f.write(str(exception))

def get_pdb_file(pdb_file, bucket):
    local_path = os.path.join(TMP_FOLDER, os.path.basename(pdb_file))
    bucket.download_file(pdb_file, local_path)
    return local_path

for pdb_file in all_pdbs:
    local_path = get_pdb_file(pdb_file, bucket)
    basename = os.path.basename(local_path)
    id = f"{basename.split('.')[0]}_{basename.split('.')[1][-1]}"
    try:
        pdb_dict = open_pdb(local_path)
    except Exception as e:
        log_exception(e, LOG_FILE, id)
        continue
    try:
        pdb_dict = align_pdb(pdb_dict)
    except Exception as e:
        log_exception(e, LOG_FILE, id)
        continue

        
    if pdb_dict is not None:
        with open(os.path.join(OUTPUT_FOLDER, id + '.pickle'), "wb") as f:
            pickle.dump(pdb_dict, f)