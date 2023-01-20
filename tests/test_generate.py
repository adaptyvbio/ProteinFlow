from bestprot import generate_data, split_data
import os
import subprocess
import pickle
from collections import defaultdict
import editdistance
import shutil


def get_class(seqs_dict):
    """Check if the protein is a homomer, a heteromer or a single chain"""
    
    keys = list(seqs_dict.keys())
    if len(keys) == 1:
        return "single_chain"
    ref_seq = list(seqs_dict.values())[0]["seq"]
    for key in keys[1:]:
        value = seqs_dict[key]
        seq = value["seq"]
        if (
            len(seq) > 1.1 * len(ref_seq)
            or len(seq) < 0.9 * len(ref_seq)
            or editdistance.eval(seq, ref_seq) / max(len(seq), len(ref_seq))
            > 0.1
        ):
            return "heteromers"
    return "homomers"

def test_generate():
    """Test generate_data + split_data + chain class distribution"""

    if os.path.exists("./data/bestprot_test"):
        shutil.rmtree("./data/bestprot_test")
    generate_data(tag="test", skip_splitting=True, n=100)
    split_data(tag="test", valid_split=0.2, test_split=0.1)
    folder = "./data/bestprot_test"
    assert os.path.exists(folder)
    assert len(os.listdir(folder)) == 4

    for subset in ["train", "valid", "test"]:
        subset_folder = os.path.join(folder, subset)
        with open(os.path.join(folder, "splits_dict", f"{subset}.pickle"), "rb") as f:
            for i in range(2):
                class_data = pickle.load(f)
        classes = defaultdict(int)
        for file in os.listdir(subset_folder):
            with open(os.path.join(subset_folder, file), "rb") as f:
                data = pickle.load(f)
            c = get_class(data)
            classes[c] += 1
        for c in classes:
            print(f'{c=}')
            print(f'{classes[c]=}')
            print(f'{sum([len(v) for v in class_data[c].values()])=}')
            assert classes[c] == sum([len(v) for v in class_data[c].values()])

    # subprocess.run(["sudo", "rm", "-rf", "data/bestprot_test"])
    
test_generate()

