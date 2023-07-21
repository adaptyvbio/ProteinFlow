"""Visualization functions for `proteinflow`."""

import py3Dmol

from proteinflow.data import PDBEntry, ProteinEntry


def show_animation_from_pdb(pdb_paths, highlight_mask_dict=None):
    """Show an animation of the given PDB files.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    highlight_mask_dict : dict, optional
        Dictionary of masks to highlight. The keys are the names of the chains, and the values are `numpy` arrays of
        1s and 0s, where 1s indicate the atoms to highlight.

    """
    entries = [PDBEntry(path) for path in pdb_paths]
    models = ""
    for i, mol in enumerate(entries):
        models += "MODEL " + str(i) + "\n"
        atoms = mol._get_atom_dicts(highlight_mask_dict=highlight_mask_dict)
        models += "".join([str(x) for x in atoms])
        models += "ENDMDL\n"

    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(models)

    for i, at in enumerate(atoms):
        default = {"cartoon": {"color": "black"}}
        view.setStyle({"model": -1, "serial": i + 1}, at.get("pymol", default))

    view.zoomTo()
    view.animate({"loop": "forward"})
    view.show()


def show_animation_from_pickle(pickle_paths, highlight_mask=None):
    """Show an animation of the given pickle files.

    Parameters
    ----------
    pickle_paths : list of str
        List of paths to pickle files.
    highlight_mask : numpy.ndarray, optional
        Mask to highlight. 1s indicate the atoms to highlight; assumes
        the chains to be concatenated in alphabetical order.

    """
    entries = [ProteinEntry.from_pickle(path) for path in pickle_paths]
    models = ""
    for i, mol in enumerate(entries):
        models += "MODEL " + str(i) + "\n"
        atoms = mol._get_atom_dicts(highlight_mask=highlight_mask)
        models += "".join([str(x) for x in atoms])
        models += "ENDMDL\n"

    view = py3Dmol.view(width=400, height=300)
    view.addModelsAsFrames(models)

    for i, at in enumerate(atoms):
        default = {"cartoon": {"color": "black"}}
        view.setStyle({"model": -1, "serial": i + 1}, at.get("pymol", default))

    view.zoomTo()
    view.animate({"loop": "forward"})
    view.show()
