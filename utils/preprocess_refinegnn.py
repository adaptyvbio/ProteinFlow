import sidechainnet as scn
import json
import numpy as np
import os
import pickle


def make_files(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for name in [
        "train",
        "valid-10",
        "valid-20",
        "valid-30",
        "valid-40",
        "valid-50",
        "valid-70",
        "valid-90",
    ]:
        print(f"Processing {name}...")
        entries = []
        for i in range(len(data[name]["seq"])):
            entry = {}
            entry["pdb"] = data[name]["ids"][i]
            entry["seq"] = data[name]["seq"][i]
            entry["cdr"] = "null"
            crd = data[name]["crd"][i].reshape((-1, 14, 3))
            entry["coords"] = {
                "N": crd[:, 0, :].tolist(),
                "CA": crd[:, 1, :].tolist(),
                "C": crd[:, 2, :].tolist(),
                "O": crd[:, 3, :].tolist(),
            }
            entries.append(entry)

        with open(os.path.join(data_path, f"{name}.jsonl"), "w") as f:
            for item in entries:
                f.write(json.dumps(item) + "\n")
