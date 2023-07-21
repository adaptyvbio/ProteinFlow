"""Provide common util functions."""
import itertools
import multiprocessing
import os
import pickle
import subprocess
import traceback
from collections import defaultdict

import numpy as np
import requests
from joblib import Parallel, delayed


class PDBError(ValueError):
    """Error raised when there is a problem with a PDB file."""

    pass


def _split_every(n, iterable):
    """Split iterable into chunks. From https://stackoverflow.com/a/1915307/2780645."""
    i = iter(iterable)
    piece = list(itertools.islice(i, n))
    while piece:
        yield piece
        piece = list(itertools.islice(i, n))


def _log_exception(exception, log_file, pdb_id, tmp_folder, chain_id=None):
    """Record the error in the log file."""
    if chain_id is None:
        _clean(pdb_id, tmp_folder)
    else:
        pdb_id = pdb_id + "-" + chain_id
    if isinstance(exception, PDBError):
        with open(log_file, "a") as f:
            f.write(f"<<< {str(exception)}: {pdb_id} \n")
    else:
        with open(log_file, "a") as f:
            f.write(f"<<< Unknown: {pdb_id} {exception}\n")
            f.write(traceback.format_exc())
            f.write("\n")


def _log_removed(removed, log_file):
    """Record which files we removed due to redundancy."""
    for pdb_id in removed:
        with open(log_file, "a") as f:
            f.write(f"<<< Removed due to redundancy: {pdb_id} \n")


def _clean(pdb_id, tmp_folder):
    """Remove all temporary files associated with a PDB ID."""
    for file in os.listdir(tmp_folder):
        if file.startswith(f"{pdb_id}."):
            subprocess.run(
                ["rm", os.path.join(tmp_folder, file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _raise_rcsbsearch(e):
    """Raise a RuntimeError if the error is due to rcsbsearch."""
    if "404 Client Error" in str(e):
        raise RuntimeError(
            'Querying rcsbsearch is failing. Please install a version of rcsbsearch where this error is solved:\npython -m pip install "rcsbsearch @ git+https://github.com/sbliven/rcsbsearch@dbdfe3880cc88b0ce57163987db613d579400c8e"'
        )
    else:
        raise e


def _make_sabdab_html(method, resolution_thr):
    """Make a URL for SAbDab search."""
    html = f"https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?ABtype=All&method={'+'.join(method)}&species=All&resolution={resolution_thr}&rfactor=&antigen=All&ltype=All&constantregion=All&affinity=All&isin_covabdab=All&isin_therasabdab=All&chothiapos=&restype=ALA&field_0=Antigens&keyword_0=#downloads"
    return html


def _test_availability(
    size_array,
    n_samples,
):
    """Test if there are enough groups in each class to construct a dataset with the required number of samples."""
    present = np.sum(size_array != 0, axis=0)
    return present[0] > n_samples, present[1] > n_samples, present[2] > n_samples


def _find_correspondences(files, dataset_dir):
    """Return a dictionary that contains all the biounits in the database (keys) and the list of all the chains that are in these biounits (values)."""
    correspondences = defaultdict(lambda: [])
    for file in files:
        biounit = file
        with open(os.path.join(dataset_dir, file), "rb") as f:
            keys = pickle.load(f)
            for k in keys:
                correspondences[biounit].append(k)

    return correspondences


def _get_number_of_chains(pdb_id):
    """Return the number of chains in a PDB file."""
    api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Extracting chain IDs
        chains = set()
        if "rcsb_entry_container_identifiers" in data:
            entity_container_identifiers = data["rcsb_entry_container_identifiers"]
            if "assembly_ids" in entity_container_identifiers:
                return len(entity_container_identifiers["assembly_ids"])

        num_chains = len(chains)

        return num_chains

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return 0


def _create_jobs(file_path, strings, results):
    """Create jobs for parallel processing."""
    # Perform your job creation logic here
    jobs = []
    for string in strings:
        for i in range(results[string]):
            jobs.append((file_path, string, i))
    return jobs


def _process_strings(strings):
    """Process strings in parallel."""
    results = {}
    processed_results = Parallel(n_jobs=-1)(
        delayed(_get_number_of_chains)(string) for string in strings
    )

    for string, result in zip(strings, processed_results):
        results[string] = result

    return results


def _write_string_to_file(file_path, string, i):
    """Write a string to a file."""
    with open(file_path, "a") as file:
        file.write(string.upper() + "-" + str(i + 1) + "\n")


def _parallel_write_to_file(file_path, jobs):
    """Write a list of strings to a file in parallel."""
    # Create a multiprocessing Pool with the desired number of processes
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Map the write_string_to_file function to each string in the list
    pool.starmap(_write_string_to_file, jobs)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    print(f"The list has been written to the file '{file_path}' successfully.")


def _write_list_to_file(file_path, string_list):
    """Write a list of strings to a file."""
    try:
        with open(file_path, "w") as file:
            # Write each string in the list to the file
            for string in string_list:
                file.write(string + "\n")  # Add a newline character after each string

        print(f"The list has been written to the file '{file_path}' successfully.")

    except OSError:
        print(f"An error occurred while writing to the file '{file_path}'.")
