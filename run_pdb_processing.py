from parse_pdb import align_pdb, open_pdb, PDBError, get_pdb_file
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
@click.option("--missing_thr", default=0.1, help="The maximum fraction of missing residues")
@click.option("--filter_methods", default=True, help="If `True`, only files obtained with X-ray or EM will be processed")
@click.option("-n", default=None, type=int, help="The number of files to process (for debugging purposes)")
@click.command()
def main(tmp_folder, output_folder, log_folder, min_length, max_length, resolution_thr, missing_thr, filter_methods, n):
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
    PDB_PREFIX = "20220103/pub/pdb/data/biounit/PDB/all/"
    MIN_LENGTH = min_length
    MAX_LENGTH = max_length
    RESOLUTION_THR = resolution_thr
    MISSING_THR = missing_thr

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

    # get filtered PDB ids
    pdb_ids = Attr('rcsb_entry_info.selected_polymer_entity_types').__eq__("Protein (only)") \
        .and_("exptl.method").in_(["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY"])
    if filter_methods:
        pdb_ids = pdb_ids.and_("rcsb_entry_info.resolution_combined").__le__(RESOLUTION_THR)
    pdb_ids = pdb_ids.exec("assembly")
    if n is not None:
        pdbs = []
        for i, x in enumerate(pdb_ids):
            pdbs.append(x)
            if i == n:
                break
        pdb_ids = pdbs

    def process_f(pdb_id, show_error=False, force=True):
        pdb_id = pdb_id.lower()
        id, biounit = pdb_id.split('-')
        target_file = os.path.join(OUTPUT_FOLDER, pdb_id + '.pickle')
        if not force and os.path.exists(target_file):
            return
        pdb_file = PDB_PREFIX + f'{id}.pdb{biounit}.gz'
        local_path = get_pdb_file(pdb_file, boto3.resource('s3').Bucket("pdbsnapshots"), tmp_folder=TMP_FOLDER)
        try:
            pdb_dict = open_pdb(
                local_path, 
                tmp_folder=TMP_FOLDER,
            )
            pdb_dict = align_pdb(pdb_dict, min_length=MIN_LENGTH, max_length=MAX_LENGTH, max_missing=MISSING_THR)
        except Exception as e:
            if show_error:
                raise e
            else:
                log_exception(e, LOG_FILE, pdb_id, TMP_FOLDER)
                pdb_dict = None
        
        if pdb_dict is not None:
            with open(target_file, "wb") as f:
                pickle.dump(pdb_dict, f)

    _ = p_map(process_f, pdb_ids)


if __name__ == "__main__":
    main()