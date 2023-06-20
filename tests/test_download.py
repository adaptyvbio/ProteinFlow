import os
import shutil
import subprocess

import pytest

from proteinflow import ProteinLoader


@pytest.mark.parametrize("tag", ["test", "test_old"])
def test_download(tag):
    """Test download_data + split_data + ProteinLoader"""

    folder = f"./data/proteinflow_{tag}"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    subprocess.run(
        ["proteinflow", "download", "--tag", tag, "--skip_splitting"], check=True
    )
    subprocess.run(["proteinflow", "split", "--tag", tag], check=True)
    for cluster_dict_path, entry_type, classes_to_exclude in [
        (os.path.join(folder, "splits_dict/valid.pickle"), "chain", ["homomers"]),
        (None, "pair", None),
    ]:
        valid_loader = ProteinLoader.from_args(
            dataset_folder=os.path.join(folder, "valid"),
            batch_size=8,
            node_features_type="chemical+sidechain_orientation+dihedral+secondary_structure+sidechain_coords",
            rewrite=True,
            clustering_dict_path=cluster_dict_path,
            classes_dict_path=os.path.join(folder, "splits_dict", "classes.pickle"),
            entry_type=entry_type,
            classes_to_exclude=classes_to_exclude,
            max_length=1000,
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
            "sidechain_coords",
            "pdb_id",
            "chain_dict",
        }
        assert batch["X"].shape == (8, batch["X"].shape[1], 4, 3)
        assert batch["S"].shape == (8, batch["X"].shape[1])
        assert batch["dihedral"].shape == (8, batch["X"].shape[1], 2)
        assert batch["masked_res"].shape == (8, batch["X"].shape[1])
        assert batch["sidechain_coords"].shape == (8, batch["X"].shape[1], 10, 3)
        assert batch["X"].shape[1] <= 1000

    shutil.rmtree(folder)


if __name__ == "__main__":
    test_download("test_old")
