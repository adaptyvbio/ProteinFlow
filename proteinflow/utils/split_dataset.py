import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import shutil


def _biounits_in_clusters_dict(clusters_dict, excluded_files=None):
    """
    Return the list of all biounit files present in clusters_dict
    """

    if len(clusters_dict) == 0:
        return np.array([])
    if excluded_files is None:
        excluded_files = []
    return np.unique(
        [
            c[0]
            for c in list(np.concatenate(list(clusters_dict.values())))
            if c[0] not in excluded_files
        ]
    )


def _split_data(
    dataset_path="./data/proteinflow_20221110/",
    excluded_files=None,
    exclude_clusters=False,
    exclude_based_on_cdr=None,
):
    """
    Rearrange files into folders according to the dataset split dictionaries at `dataset_path/splits_dict`

    Parameters
    ----------
    dataset_path : str, default "./data/proteinflow_20221110/"
        The path to the dataset folder containing pre-processed entries and a `splits_dict` folder with split dictionaries (downloaded or generated with `get_split_dictionaries`)
    exculded_files : list, optional
        A list of files to exclude from the dataset
    exclude_clusters : bool, default False
        If True, exclude all files in a cluster if at least one file in the cluster is in `excluded_files`
    exclude_based_on_cdr : str, optional
        If not `None`, exclude all files in a cluster if the cluster name does not end with `exclude_based_on_cdr`
    """

    if excluded_files is None:
        excluded_files = []

    dict_folder = os.path.join(dataset_path, "splits_dict")
    with open(os.path.join(dict_folder, "train.pickle"), "rb") as f:
        train_clusters_dict = pkl.load(f)
    with open(os.path.join(dict_folder, "valid.pickle"), "rb") as f:
        valid_clusters_dict = pkl.load(f)
    with open(os.path.join(dict_folder, "test.pickle"), "rb") as f:
        test_clusters_dict = pkl.load(f)

    train_biounits = _biounits_in_clusters_dict(train_clusters_dict, excluded_files)
    valid_biounits = _biounits_in_clusters_dict(valid_clusters_dict, excluded_files)
    test_biounits = _biounits_in_clusters_dict(test_clusters_dict, excluded_files)
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    if len(excluded_files) > 0:
        excluded_set = set(excluded_files)
        if exclude_clusters:
            for cluster, files in train_clusters_dict.items():
                exclude = False
                for biounit in files:
                    if biounit in excluded_set:
                        exclude = True
                        break
                if exclude:
                    if exclude_based_on_cdr is not None:
                        if not cluster.endswith(exclude_based_on_cdr):
                            continue
                    for biounit in files:
                        excluded_files.append(biounit)
        excluded_path = os.path.join(dataset_path, "excluded")
        if not os.path.exists(excluded_path):
            os.makedirs(excluded_path)
        print("Moving excluded files...")
        for biounit in tqdm(excluded_files):
            shutil.move(os.path.join(dataset_path, biounit), excluded_path)
    print("Moving files in the train set...")
    for biounit in tqdm(train_biounits):
        shutil.move(os.path.join(dataset_path, biounit), train_path)
    print("Moving files in the validation set...")
    for biounit in tqdm(valid_biounits):
        shutil.move(os.path.join(dataset_path, biounit), valid_path)
    print("Moving files in the test set...")
    for biounit in tqdm(test_biounits):
        shutil.move(os.path.join(dataset_path, biounit), test_path)
