import os
import argparse
import subprocess
import numpy as np
import pickle as pkl




def biounits_in_clusters_dict(clusters_dict):

    """
    Return the list of all biounit files present in clusters_dict
    """

    return np.unique([c[0] for c in list(np.concatenate(clusters_dict.values()))])


def download_dataset_dicts_from_s3(dict_folder_path, s3_path='s3://ml4-main-storage/bestprot_20221110_splits_dict/'):

    """
    Download dictionaries containing database split information from s3 to a local folder
    """

    train_path = os.path.join(s3_path, 'train.pickle')
    valid_path = os.path.join(s3_path, 'valid.pickle')
    test_path = os.path.join(s3_path, 'test.pickle')

    if not os.path.exists(dict_folder_path):
        os.makedirs(dict_folder_path)
    
    subprocess.run(['aws', 's3', 'cp', train_path, dict_folder_path])
    subprocess.run(['aws', 's3', 'cp', valid_path, dict_folder_path])
    subprocess.run(['aws', 's3', 'cp', test_path, dict_folder_path])


def download_dataset_from_s3(train_clusters_dict, valid_clusters_dict, test_clusters_dict, dataset_path='bestprot_dataset', s3_path='s3://ml4-main-storage/bestprot_20221110/'):

    """
    Download all biounits in the database and puts them into train, validation and test folders according to the partition given by the train/valid/test_clusters_dict
    """

    train_biounits = biounits_in_clusters_dict(train_clusters_dict)
    valid_biounits = biounits_in_clusters_dict(valid_clusters_dict)
    test_biounits = biounits_in_clusters_dict(test_clusters_dict)
    train_path = os.path.join(dataset_path, 'training')
    valid_path = os.path.join(dataset_path, 'validation')
    test_path = os.path.join(dataset_path, 'test')

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    for biounit in train_biounits:
        subprocess.run(['aws', 's3', 'cp', os.path.join(s3_path, biounit), train_path])
    for biounit in valid_biounits:
        subprocess.run(['aws', 's3', 'cp', os.path.join(s3_path, biounit), valid_path])
    for biounit in test_biounits:
        subprocess.run(['aws', 's3', 'cp', os.path.join(s3_path, biounit), test_path])


def main():

    parser = argparse.ArgumentParser(description="Download dataset from s3 instance")
    
    parser.add_argument("--dataset_path", default="s3://ml4-main-storage/bestprot_20221110/", type=str, help="path to the dataset stored in the s3 instance.")
    parser.add_argument("--dict_path", default="s3://ml4-main-storage/bestprot_20221110_splits_dict/", type=str, help="path to the folder with the dataset partioning information.")
    parser.add_argument("--local_folder", default="bestprot_dataset", type=str, help="path to the folder where to store the splitted dataset.")

    args = parser.parse_args()

    S3_DATASET_PATH = args.dataset_path
    S3_DICT_PATH = args.dict_path
    DATASET_FOLDER = args.local_folder
    DICT_FOLDER = os.path.join(args.local_folder, 'splits_dict')
    
    download_dataset_dicts_from_s3(DICT_FOLDER, S3_DICT_PATH)

    with open(os.path.join(DICT_FOLDER, 'train.pickle'), 'rb') as f:
        train_clusters_dict = pkl.load(f)
    with open(os.path.join(DICT_FOLDER, 'valid.pickle'), 'rb') as f:
        valid_clusters_dict = pkl.load(f)
    with open(os.path.join(DICT_FOLDER, 'test.pickle'), 'rb') as f:
        test_clusters_dict = pkl.load(f)
    
    download_dataset_from_s3(train_clusters_dict, valid_clusters_dict, test_clusters_dict, dataset_path=DATASET_FOLDER, s3_path=S3_DATASET_PATH)


if __name__=='__main__':
    main()