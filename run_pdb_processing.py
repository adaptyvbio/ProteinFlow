from utils.filter_database import remove_database_redundancies
from utils.parse_pdb import align_pdb, open_pdb, PDBError, get_pdb_file, s3list
from utils.cluster_and_partition import build_dataset_partition
import os
import boto3
import pickle
from p_tqdm import p_map
from rcsbsearch import Attr
import subprocess
import click
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import shutil


def _clean(pdb_id, tmp_folder):
    """
    Remove all temporary files associated with a PDB ID
    """

    for file in os.listdir(tmp_folder):
        if file.startswith(f'{pdb_id}.'):
            subprocess.run(["rm", os.path.join(tmp_folder, file)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _log_exception(exception, log_file, pdb_id, tmp_folder):
    """
    Record the error in the log file
    """

    _clean(pdb_id, tmp_folder)
    if isinstance(exception, PDBError):
        with open(log_file, "a") as f:
            f.write(f'<<< {str(exception)}: {pdb_id} \n')
    else:
        with open(log_file, "a") as f:
            f.write(f'<<< Unknown: {pdb_id} \n')
            f.write(str(exception))
            f.write("\n")

def _log_removed(removed, log_file):
    """
    Record which files we removed due to redundancy
    """

    for pdb_id in removed:
        with open(log_file, "a") as f:
            f.write(f'<<< Removed due to redundancy: {pdb_id} \n')

def get_error_summary(log_file, verbose=True):
    """
    Get a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    Parameters
    ----------
    log_file : str
        the log file path
    verbose : bool, default True
        if `True`, the statistics are written in the standard output

    Returns
    -------
    log_dict : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors

    """
    stats = defaultdict(lambda: [])
    with open(log_file, "r") as f:
        for line in f.readlines():
            if line.startswith("<<<"):
                stats[line.split(':')[0]].append(line.split(":")[-1].strip())
    keys = sorted(stats.keys(), key=lambda x: stats[x], reverse=True)
    if verbose:
        for key in keys:
            print(f'{key}: {len(stats[key])}')
    return stats

def run_processing(
        tmp_folder="./data/tmp_pdb", 
        output_folder="./data/pdb", 
        log_folder="./data/logs", 
        min_length=30, 
        max_length=10000, 
        resolution_thr=3.5, 
        missing_ends_thr=0.3, 
        missing_middle_thr=0.1, 
        filter_methods=True, 
        remove_redundancies=False, 
        seq_identity_threshold=0.9, 
        n=None, 
        force=False,
        split_tolerance=0.2,
        split_database=False,
        test_split=0.05,
        valid_split=0.05,
        out_split_dict_folder="./data/dataset_splits_dicts",
    ):
    """
    Download and parse PDB files that meet filtering criteria

    The output files are pickled nested dictionaries where first-level keys are chain Ids and second-level keys are 
    the following:
    
    - `'crd_bb'`: a `numpy` array of shape `(L, 4, 3)` with backbone atom coordinates (N, C, CA, O),
    - `'crd_sc'`: a `numpy` array of shape `(L, 10, 3)` with sidechain atom coordinates (in a fixed order),
    - `'msk'`: a `numpy` array of shape `(L,)` where ones correspond to residues with known coordinates and 
        zeros to missing values,
    - `'seq'`: a string of length `L` with residue types.

    All errors including reasons for filtering a file out are logged in the log file.

    Parameters
    ----------
    tmp_folder : str, default "./data/tmp_pdb"
        The folder where temporary files will be saved
    output_folder : str, default "./data/pdb"
        The folder where the output files will be saved
    log_folder : str, default "./data/logs"
        The folder where the log file will be saved
    min_length : int, default 30
        The minimum number of non-missing residues per chain
    max_length : int, default 10000
        The maximum number of residues per chain (set None for no threshold)
    resolution_thr : float, default 3.5
        The maximum resolution
    missing_ends_thr : float, default 0.3
        The maximum fraction of missing residues at the ends
    missing_middle_thr : float, default 0.1
        The maximum fraction of missing residues in the middle (after missing ends are disregarded)
    filter_methods : bool, default True
        If `True`, only files obtained with X-ray or EM will be processed
    remove_redundancies : bool, default False
        If 'True', removes biounits that are doubles of others sequence wise
    seq_identity_threshold : float, default 0.9
        The threshold upon which sequences are considered as one and the same (default: 90%)
    n : int, default None
        The number of files to process (for debugging purposes)
    force : bool, default False
        When `True`, rewrite the files if they already exist
    split_tolerance : float, default 0.2
        The tolerance on the split ratio (default 20%)
    split_database : bool, default False
        Whether or not to split the database
    test_split : float, default 0.05
        The percentage of chains to put in the test set (default 5%)
    valid_split : float, default 0.05
        The percentage of chains to put in the validation set (default 5%)
    out_split_dict_folder : str, default "./data/dataset_splits_dicts"
        The folder where the dictionaries containing the train/validation/test splits information will be saved"

    Returns
    -------
    log : dict
        a dictionary where keys are recognized error names and values are lists of PDB ids that caused the errors       

    """
    TMP_FOLDER = tmp_folder
    OUTPUT_FOLDER = output_folder
    DICT_FOLDER = out_split_dict_folder
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
        .or_('rcsb_entry_info.polymer_composition').__eq__("protein/oligosaccharide")
    
    pdb_ids = pdb_ids.and_("rcsb_entry_info.resolution_combined").__le__(RESOLUTION_THR)
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
    ordered_folders = sorted(ordered_folders, reverse=True) # a list of PDB snapshots from newest to oldest

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
                _log_exception(e, LOG_FILE, pdb_id, TMP_FOLDER)

    # process_f("1a1q-3", show_error=True, force=force)

    _ = p_map(lambda x: process_f(x, force=force), pdb_ids)
    
    stats = get_error_summary(LOG_FILE, verbose=False)
    while "<<< PDB file not found" in stats:
        with open(f'{LOG_FILE}', "r") as f:
            lines = [x for x in f.readlines() if not x.startswith("<<< PDB file not found")]
        os.remove(f'{LOG_FILE}')
        with open(f'{LOG_FILE}_tmp', "a") as f:
            for line in lines:
                f.write(line)
        _ = p_map(lambda x: process_f(x, force=force), stats["<<< PDB file not found"])
        stats = get_error_summary(LOG_FILE, verbose=False)
    if os.path.exists(f'{LOG_FILE}_tmp'):
        with open(f'{LOG_FILE}', "r") as f:
            lines = [x for x in f.readlines() if not x.startswith("<<< PDB file not found")]
        os.remove(f'{LOG_FILE}')
        with open(f'{LOG_FILE}_tmp', "a") as f:
            for line in lines:
                f.write(line)
        os.rename(f'{LOG_FILE}_tmp', LOG_FILE)

    if remove_redundancies:
        removed = remove_database_redundancies(OUTPUT_FOLDER, seq_identity_threshold=seq_identity_threshold)
        _log_removed(removed, LOG_FILE)
    
    if split_database:
        (
            train_clusters_dict,
            train_classes_dict,
            valid_clusters_dict,
            valid_classes_dict,
            test_clusters_dict,
            test_classes_dict,
        ) = build_dataset_partition(OUTPUT_FOLDER, TMP_FOLDER, valid_split=valid_split, test_split=test_split, tolerance=split_tolerance)
        with open(os.path.join(DICT_FOLDER, 'train.pickle'), 'wb') as f:
            pickle.dump(train_clusters_dict, f)
            pickle.dump(train_classes_dict, f)
        with open(os.path.join(DICT_FOLDER, 'valid.pickle'), 'wb') as f:
            pickle.dump(valid_clusters_dict, f)
            pickle.dump(valid_classes_dict, f)
        with open(os.path.join(DICT_FOLDER, 'test.pickle'), 'wb') as f:
            pickle.dump(test_clusters_dict, f)
            pickle.dump(test_classes_dict, f)
    
    return get_error_summary(LOG_FILE)


@click.option("--tmp_folder", default="./data/tmp_pdb", help="The folder where temporary files will be saved")
@click.option("--output_folder", default="./data/pdb", help="The folder where the output files will be saved")
@click.option("--log_folder", default="./data/logs", help="The folder where the log file will be saved")
@click.option("--out_split_dict_folder", default="./data/dataset_splits_dicts", help="The folder where the dictionaries containing the train/validation/test splits information will be saved")
@click.option("--min_length", default=30, help="The minimum number of non-missing residues per chain")
@click.option("--max_length", default=10000, help="The maximum number of residues per chain (set None for no threshold)")
@click.option("--resolution_thr", default=3.5, help="The maximum resolution")
@click.option("--missing_ends_thr", default=0.3, help="The maximum fraction of missing residues at the ends")
@click.option("--missing_middle_thr", default=0.1, help="The maximum fraction of missing residues in the middle (after missing ends are disregarded)")
@click.option("--filter_methods", default=True, help="If `True`, only files obtained with X-ray or EM will be processed")
@click.option("--remove_redundancies", default=False, help="If 'True', removes biounits that are doubles of others sequence wise")
@click.option("--seq_identity_threshold", default=.9, type=float, help="The threshold upon which sequences are considered as one and the same (default: 90%)")
@click.option("--split_database", default=False, help="Whether or not to split the database")
@click.option("--valid_split", default=.05, type=float, help="The percentage of chains to put in the validation set (default 5%)")
@click.option("--test_split", default=.05, type=float, help="The percentage of chains to put in the test set (default 5%)")
@click.option("--split_tolerance", default=.2, type=float, help="The tolerance on the split ratio (default 20%)")
@click.option("-n", default=None, type=int, help="The number of files to process (for debugging purposes)")
@click.option("--force", is_flag=True, help="When `True`, rewrite the files if they already exist")
@click.command()
def main(
        tmp_folder="./data/tmp_pdb", 
        output_folder="./data/pdb", 
        log_folder="./data/logs", 
        min_length=30, 
        max_length=10000, 
        resolution_thr=3.5, 
        missing_ends_thr=0.3, 
        missing_middle_thr=0.1, 
        filter_methods=True, 
        remove_redundancies=False, 
        seq_identity_threshold=0.9, 
        n=None, 
        force=False,
        split_tolerance=0.2,
        split_database=False,
        test_split=0.05,
        valid_split=0.05,
        out_split_dict_folder="./data/dataset_splits_dicts",
    ):
    run_processing(
        tmp_folder, 
        output_folder, 
        log_folder, 
        min_length, 
        max_length, 
        resolution_thr, 
        missing_ends_thr, 
        missing_middle_thr, 
        filter_methods, 
        remove_redundancies, 
        seq_identity_threshold, 
        n, 
        force,
        split_tolerance,
        split_database,
        test_split,
        valid_split,
        out_split_dict_folder,
    )

if __name__ == "__main__":
    main()