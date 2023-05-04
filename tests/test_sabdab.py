from proteinflow import generate_data, split_data
import os
import pickle
from collections import defaultdict
import editdistance
import shutil
from time import time
import pytest


# @pytest.mark.skip()
def test_generate_sabdab():
    """Test generate_data with `sabdab=True`"""

    folder = "./data/proteinflow_test"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    start = time()
    generate_data(tag="test", skip_splitting=True, n=50, sabdab=True, resolution_thr=1)
    end = time()
    shutil.rmtree(folder)
    print(f"generation time: {end - start} sec")
    start = time()
    generate_data(tag="test", skip_splitting=True, sabdab=True, zip_path="./sample_data/sabdab.zip", require_antigen=True)
    end = time()
    assert all(["nan_nan" not in file for file in os.listdir(folder)])
    shutil.rmtree(folder)
    print(f"generation time: {end - start} sec")

if __name__ == "__main__":
    test_generate_sabdab()
