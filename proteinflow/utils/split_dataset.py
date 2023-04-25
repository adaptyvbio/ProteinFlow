import os
import subprocess
import numpy as np
import pickle as pkl
from tqdm import tqdm
import shutil


def _get_s3_paths_from_tag(tag):
    """
    Get the path to the data and split dictionary folders on S3 given a tag
    """

    dict_path = f"s3://proteinflow-datasets/{tag}/proteinflow_{tag}_splits_dict/"
    data_path = f"s3://proteinflow-datasets/{tag}/proteinflow_{tag}/"
    return data_path, dict_path


def _biounits_in_clusters_dict(clusters_dict, excluded_files=None):
    """
    Return the list of all biounit files present in clusters_dict
    """

    if excluded_files is None:
        excluded_files = []
    return np.unique(
        [
            c[0]
            for c in list(np.concatenate(list(clusters_dict.values())))
            if c[0] not in excluded_files
        ]
    )


def _download_dataset_dicts_from_s3(dict_folder_path, s3_path):
    """
    Download dictionaries containing database split information from s3 to a local folder
    """

    train_path = os.path.join(s3_path, "train.pickle")
    valid_path = os.path.join(s3_path, "valid.pickle")
    test_path = os.path.join(s3_path, "test.pickle")

    if not os.path.exists(dict_folder_path):
        os.makedirs(dict_folder_path)

    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", train_path, dict_folder_path]
    )
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", valid_path, dict_folder_path]
    )
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", test_path, dict_folder_path]
    )


def _split_data(dataset_path="./data/proteinflow_20221110/", excluded_files=None):
    """
    Rearrange files into folders according to the dataset split dictionaries at `dataset_path/splits_dict`

    Parameters
    ----------
    dataset_path : str, default "./data/proteinflow_20221110/"
        The path to the dataset folder containing pre-processed entries and a `splits_dict` folder with split dictionaries (downloaded or generated with `get_split_dictionaries`)
    exculded_files : list, optional
        A list of files to exclude from the dataset
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


def _download_dataset_from_s3(
    dataset_path="./data/proteinflow_20221110/",
    s3_path="s3://ml4-main-storage/proteinflow_20221110/",
):
    """
    Download the pre-processed files
    """

    if s3_path.startswith("s3"):
        print("Downloading the dataset from s3...")
        subprocess.run(
            ["aws", "s3", "sync", "--no-sign-request", s3_path, dataset_path]
        )
        print("Done!")
    else:
        shutil.move(s3_path, dataset_path)


def _download_dataset(tag, local_datasets_folder="./data/"):
    """
    Download the pre-processed data and the split dictionaries

    Parameters
    ----------
    tag : str
        name of the dataset (check `get_available_tags` to see the options)
    local_dataset_folder : str, default "./data/"
        the local folder that will contain proteinflow dataset folders, temporary files and logs

    Returns
    -------
    data_folder : str
        the path to the downloaded data folder
    """

    s3_data_path, s3_dict_path = _get_s3_paths_from_tag(tag)
    data_folder = os.path.join(local_datasets_folder, f"proteinflow_{tag}")
    dict_folder = os.path.join(
        local_datasets_folder, f"proteinflow_{tag}", "splits_dict"
    )

    print("Downloading dictionaries for splitting the dataset...")
    _download_dataset_dicts_from_s3(dict_folder, s3_dict_path)
    print("Done!")

    _download_dataset_from_s3(dataset_path=data_folder, s3_path=s3_data_path)
    return data_folder
