"""Functions for logging errors and warnings."""

import os
import subprocess
import traceback
from collections import defaultdict

from proteinflow.data.utils import PDBError


def get_error_summary(log_file, verbose=True):
    """Get an exception summary.

    The output is a dictionary where keys are recognized exception names and values are lists of
    PDB ids that caused the exceptions.

    Parameters
    ----------
    log_file : str
        the log file path
    verbose : bool, default True
        if `True`, the statistics are written in the standard output

    Returns
    -------
    log_dict : dict
        a dictionary where keys are recognized exception names and values are lists of PDB ids that
        caused the exceptions

    """
    stats = defaultdict(lambda: [])
    with open(log_file) as f:
        for line in f.readlines():
            if line.startswith("<<<"):
                stats[line.split(":")[0]].append(line.split(":")[-1].strip())
    if verbose:
        keys = sorted(stats.keys(), key=lambda x: len(stats[x]), reverse=True)
        for key in keys:
            print(f"{key}: {len(stats[key])}")
        print(f"Total exceptions: {sum([len(x) for x in stats.values()])}")
    return stats


def _clean(pdb_id, tmp_folder):
    """Remove all temporary files associated with a PDB ID."""
    for file in os.listdir(tmp_folder):
        if file.startswith(f"{pdb_id}."):
            subprocess.run(
                ["rm", os.path.join(tmp_folder, file)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


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
