import os
import argparse
import subprocess
import numpy as np
import pickle as pkl
from tqdm import tqdm
import shutil


def biounits_in_clusters_dict(clusters_dict):

    """
    Return the list of all biounit files present in clusters_dict
    """

    return np.unique([c[0] for c in list(np.concatenate(list(clusters_dict.values())))])

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

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    if s3_path.startswith("s3"):
        print('Downloading the datset from s3...')
        subprocess.run(['aws', 's3', 'cp', '--recursive', s3_path, dataset_path])
        print('Done!')
        print("We're almost there, just a tiny effort left :-)")
    else:
        shutil.move(s3_path, dataset_path)

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    print('Moving files in the train set...')
    for biounit in tqdm(train_biounits):
        subprocess.run(['mv', os.path.join(dataset_path, biounit), train_path])
    print('Done!')
    print('Moving files in the validation set...')
    for biounit in tqdm(valid_biounits):
        subprocess.run(['mv', os.path.join(dataset_path, biounit), valid_path])
    print('Done!')
    print('Moving files in the test set...')
    for biounit in tqdm(test_biounits):
        subprocess.run(['mv', os.path.join(dataset_path, biounit), test_path])
    print('Done!')
    print('-------------------------------------')
    print('Thanks for downloading BestProt, the most complete, user-friendly and loving protein dataset you will ever find! ;-)')

def main():

    parser = argparse.ArgumentParser(description="Download dataset from s3 instance")
    
    parser.add_argument("--dataset_path", default="s3://ml4-main-storage/bestprot_20221110/", type=str, help="path to the dataset, either local or in the s3 instance.")
    parser.add_argument("--dict_path", default="s3://ml4-main-storage/bestprot_20221110_splits_dict/", type=str, help="path to the folder with the dataset partioning information, either local or in the s3 instance.")
    parser.add_argument("--local_folder", default="bestprot_dataset", type=str, help="path to the folder where to store the splitted dataset (if dataset_path is local, the folder will be moved here).")

    args = parser.parse_args()

    S3_DATASET_PATH = args.dataset_path
    S3_DICT_PATH = args.dict_path
    DATASET_FOLDER = args.local_folder
    DICT_FOLDER = os.path.join(args.local_folder, 'splits_dict')
    
    if S3_DICT_PATH.startswith("s3"):
        print('Downloading dictionaries for splitting the dataset...')
        download_dataset_dicts_from_s3(DICT_FOLDER, S3_DICT_PATH)
        print('Done!')
    else:
        shutil.move(S3_DICT_PATH, DICT_FOLDER)

    with open(os.path.join(DICT_FOLDER, 'train.pickle'), 'rb') as f:
        train_clusters_dict = pkl.load(f)
    with open(os.path.join(DICT_FOLDER, 'valid.pickle'), 'rb') as f:
        valid_clusters_dict = pkl.load(f)
    with open(os.path.join(DICT_FOLDER, 'test.pickle'), 'rb') as f:
        test_clusters_dict = pkl.load(f)
    
    download_dataset_from_s3(train_clusters_dict, valid_clusters_dict, test_clusters_dict, dataset_path=DATASET_FOLDER, s3_path=S3_DATASET_PATH)


if __name__=='__main__':
    main()