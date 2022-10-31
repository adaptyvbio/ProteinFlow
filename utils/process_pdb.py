from utils.parse_pdb import align_pdb, open_pdb
import os
import boto3
import pickle

TMP_FOLDER = "data/tmp_pdb"
OUTPUT_FOLDER = "data/pdb"
PDB_PREFIX = "20220103/pub/pdb/data/biounit/PDB/all/"

if not os.path.exists(TMP_FOLDER):
    os.mkdir(TMP_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

s3 = boto3.resource('s3')
bucket = s3.Bucket("pdbsnapshots")
all_pdbs = bucket.objects.filter(Prefix=PDB_PREFIX)[:10]

def get_pdb_file(pdb_file, bucket):
    local_path = os.path.join(TMP_FOLDER, os.path.basename(pdb_file))
    bucket.download_file(pdb_file, local_path)
    return local_path

for pdb_file in all_pdbs:
    local_path = get_pdb_file(pdb_file, bucket)
    basename = os.path.basename(local_path)
    id = f"{basename.split('.')[0]}_{basename.split('.')[1][-1]}"
    pdb_dict = open_pdb(local_path)
    pdb_dict = align_pdb(pdb_dict)
    if pdb_dict is not None:
        with open(os.path.join(OUTPUT_FOLDER, id + '.pickle'), "wb") as f:
            pickle.dump(pdb_dict, f)