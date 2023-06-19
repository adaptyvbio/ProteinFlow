import os
import pickle
import shutil
from collections import defaultdict
from time import time

import editdistance

from proteinflow import generate_data, split_data


def get_class(seqs_dict):
    """Check if the protein is a homomer, a heteromer or a single chain"""

    keys = list(seqs_dict.keys())
    if len(keys) == 1:
        return "single_chains"
    ref_seq = list(seqs_dict.values())[0]["seq"]
    for key in keys[1:]:
        value = seqs_dict[key]
        seq = value["seq"]
        if (
            len(seq) > 1.1 * len(ref_seq)
            or len(seq) < 0.9 * len(ref_seq)
            or editdistance.eval(seq, ref_seq) / max(len(seq), len(ref_seq)) > 0.1
        ):
            return "heteromers"
    return "homomers"


# @pytest.mark.skip()
def test_generate():
    """Test generate_data + split_data + chain class distribution"""

    folder = "./data/proteinflow_test"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    start = time()
    generate_data(tag="test", skip_splitting=True, n=50)
    end = time()
    num_files = len(os.listdir(folder))
    split_data(
        tag="test",
        valid_split=0.2,
        test_split=0.1,
        ignore_existing=True,
        exclude_chains=["1arx-A"],
    )
    assert os.path.exists(folder)
    assert len(os.listdir(folder)) == 6
    num_files_split = 0

    with open(
        os.path.join(folder, "splits_dict", "classes.pickle"),
        "rb",
    ) as f:
        class_data = pickle.load(f)
    classes = defaultdict(int)
    for subset in ["train", "valid", "test", "excluded"]:
        num_files_split += len(os.listdir(os.path.join(folder, subset)))
        subset_folder = os.path.join(folder, subset)
        for file in os.listdir(subset_folder):
            with open(os.path.join(subset_folder, file), "rb") as f:
                data = pickle.load(f)
            c = get_class(data)
            classes[c] += 1
    for c in class_data:
        class_files = set()
        for chain_arr in class_data[c].values():
            for file, _ in chain_arr:
                class_files.add(file)
        assert classes[c] == len(class_files)
    assert num_files == num_files_split + 1
    shutil.rmtree(folder)
    print(f"generation time: {end - start} sec")


if __name__ == "__main__":
    test_generate()
