from filter_database import remove_database_redundancies
from utils.parse_pdb import align_pdb, open_pdb, PDBError, get_pdb_file, s3list
import os
import boto3
import pickle
from p_tqdm import p_map
from rcsbsearch import Attr
import subprocess
import click
from datetime import datetime
from tqdm import tqdm


def clean(pdb_id, tmp_folder):
    """
    Remove all temporary files associated with a PDB ID
    """

    for file in os.listdir(tmp_folder):
        if file.startswith(pdb_id):
            subprocess.run(["rm", os.path.join(tmp_folder, file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def log_exception(exception, log_file, pdb_id, tmp_folder):
    """
    Record the error in the log file
    """

    clean(pdb_id, tmp_folder)
    if isinstance(exception, PDBError):
        with open(log_file, "a") as f:
            f.write(f'<<< {str(exception)}: {pdb_id} \n')
    else:
        with open(log_file, "a") as f:
            f.write(f'<<< Unknown: {pdb_id} \n')
            f.write(str(exception))
            f.write("\n")


@click.option("--tmp_folder", default="./data/tmp_pdb", help="The folder where temporary files will be saved")
@click.option("--output_folder", default="./data/pdb", help="The folder where the output files will be saved")
@click.option("--log_folder", default="./data/logs", help="The folder where the log file will be saved")
@click.option("--min_length", default=30, help="The minimum number of non-missing residues per chain")
@click.option("--max_length", default=10000, help="The maximum number of residues per chain (set None for no threshold)")
@click.option("--resolution_thr", default=3.5, help="The maximum resolution")
@click.option("--missing_ends_thr", default=0.3, help="The maximum fraction of missing residues at the ends")
@click.option("--missing_middle_thr", default=0.1, help="The maximum fraction of missing residues in the middle (after missing ends are disregarded)")
@click.option("--filter_methods", default=True, help="If `True`, only files obtained with X-ray or EM will be processed")
@click.option("--remove_redundancies", default=True, help="If 'True', removes biounits that are doubles of others sequence wise")
@click.option("--seq_identity_threshold", default=.9, type=float, help="The threshold upon which sequences are considered as one and the same (default: 90%)")
@click.option("-n", default=None, type=int, help="The number of files to process (for debugging purposes)")
@click.option("--force", default=False, help="When `True`, rewrite the files if they already exist")
@click.command()
def main(tmp_folder, output_folder, log_folder, min_length, max_length, resolution_thr, missing_ends_thr, missing_middle_thr, filter_methods, remove_redundancies, seq_identity_threshold, n, force):
    """
    Download and parse PDB files that meet filtering criteria

    The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are 
    the following:
    
    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, Ca, C, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order),
    - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and 
        zeros to missing values,
    - `'seq'`: a string of length `L` with residue types.

    All errors including reasons for filtering a file out are logged in the log file.
    """

    TMP_FOLDER = tmp_folder
    OUTPUT_FOLDER = output_folder
    PDB_PREFIX = "pub/pdb/data/biounit/PDB/all/"
    MIN_LENGTH = min_length
    MAX_LENGTH = max_length
    RESOLUTION_THR = resolution_thr
    MISSING_ENDS_THR = missing_ends_thr
    MISSING_MIDDLE_THR = missing_middle_thr

    if not os.path.exists(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    i = 0
    while os.path.exists(os.path.join(log_folder, f"log_{i}.txt")):
        i += 1
    LOG_FILE = os.path.join(log_folder, f"log_{i}.txt")
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n\n"
    with open(LOG_FILE, "a") as f:
        f.write(date_time)

    # get filtered PDB ids fro PDB API
    pdb_ids = Attr('rcsb_entry_info.selected_polymer_entity_types').__eq__("Protein (only)") \
        .and_("rcsb_entry_info.resolution_combined").__le__(RESOLUTION_THR)
    if filter_methods:
        pdb_ids = pdb_ids.and_("exptl.method").in_(["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"])
    pdb_ids = pdb_ids.exec("assembly")
    if n is not None:
        pdbs = []
        for i, x in enumerate(pdb_ids):
            pdbs.append(x)
            if i == n:
                break
        pdb_ids = pdbs
    
    ordered_folders = [x.key + PDB_PREFIX for x in s3list(boto3.resource('s3').Bucket("pdbsnapshots"), "", recursive=False, list_objs=False)]
    ordered_folders = sorted(ordered_folders, reverse=True)

    def process_f(pdb_id, show_error=False, force=True):
        try:
            pdb_id = pdb_id.lower()
            id, biounit = pdb_id.split('-')
            target_file = os.path.join(OUTPUT_FOLDER, pdb_id + '.pickle')
            if not force and os.path.exists(target_file):
                raise PDBError("File already exists")
            pdb_file = f'{id}.pdb{biounit}.gz'
            # download
            local_path = get_pdb_file(
                pdb_file, 
                boto3.resource('s3').Bucket("pdbsnapshots"), 
                tmp_folder=TMP_FOLDER, 
                folders=ordered_folders
            )
            # parse
            pdb_dict = open_pdb(
                local_path, 
                tmp_folder=TMP_FOLDER,
            )
            # filter and convert
            pdb_dict = align_pdb(
                pdb_dict, 
                min_length=MIN_LENGTH, 
                max_length=MAX_LENGTH, 
                max_missing_ends=MISSING_ENDS_THR,
                max_missing_middle=MISSING_MIDDLE_THR,
            )
            # save
            if pdb_dict is not None:
                with open(target_file, "wb") as f:
                    pickle.dump(pdb_dict, f)
        except Exception as e:
            if show_error:
                raise e
            else:
                log_exception(e, LOG_FILE, pdb_id, TMP_FOLDER)

    _ = p_map(lambda x: process_f(x, force=force), pdb_ids)
    # process_f("1b79-1", show_error=True)

    if remove_redundancies:
        remove_database_redundancies(OUTPUT_FOLDER, seq_identity_threshold=seq_identity_threshold)


if __name__ == "__main__":
    main()