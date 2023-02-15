import subprocess
import shutil
from proteinflow import ProteinLoader
import os
 

def test_download():
    """Test download_data + split_data + ProteinLoader"""

    folder = "./data/proteinflow_test"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    subprocess.run(
        ["proteinflow", "download", "--tag", "test", "--skip_splitting"], check=True
    )
    subprocess.run(["proteinflow", "split", "--tag", "test"], check=True)
    for cluster_dict_path, entry_type, classes_to_exclude in [
        (None, "chain", None),
        (os.path.join(folder, "splits_dict/valid.pickle"), "pair", ["homomers"]),
    ]:
        valid_loader = ProteinLoader(
            os.path.join(folder, "valid"),
            batch_size=8,
            node_features_type="chemical+sidechain_orientation+dihedral+secondary_structure",
            rewrite=True,
            clustering_dict_path=cluster_dict_path,
            entry_type=entry_type,
            classes_to_exclude=classes_to_exclude,
        )
        batch = next(iter(valid_loader))
        assert set(batch.keys()) == {
            "X",
            "S",
            "mask",
            "mask_original",
            "residue_idx",
            "chain_encoding_all",
            "chain_id",
            "sidechain_orientation",
            "dihedral",
            "chemical",
            "secondary_structure",
            "masked_res",
        }
        assert batch["X"].shape == (8, batch["X"].shape[1], 4, 3)
        assert batch["S"].shape == (8, batch["X"].shape[1])
        assert batch["dihedral"].shape == (8, batch["X"].shape[1], 2)
        assert batch["masked_res"].shape == (8, batch["X"].shape[1])

    shutil.rmtree(folder)


# test_download()
