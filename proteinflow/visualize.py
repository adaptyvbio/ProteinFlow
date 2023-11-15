"""Visualization functions for `proteinflow`."""

import string

import numpy as np

from proteinflow.data import PDBEntry, ProteinEntry
from proteinflow.extra import _get_view


def show_animation_from_pdb(
    pdb_paths,
    highlight_mask_dict=None,
    style="cartoon",
    opacity=1,
    direction="forward",
    colors=None,
    accent_color="#D96181",
    canvas_size=(400, 300),
):
    """Show an animation of the given PDB files.

    Parameters
    ----------
    pdb_paths : list of str
        List of paths to PDB files.
    highlight_mask_dict : dict, optional
        Dictionary of masks to highlight. The keys are the names of the chains, and the values are `numpy` arrays of
        1s and 0s, where 1s indicate the atoms to highlight.
    style : str, optional
        The style of the visualization; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'
    opacity : float, default 1
        The opacity of the visualization.
    direction : {"forward", "backAndForth"}
        The direction of the animation.
    colors : list of str, optional
        List of colors to use for each chain
    accent_color : str, optional
        Color to use for the highlighted atoms (use `None` to disable)
    canvas_size : tuple of int, optional
        The size of the canvas to display the animation on

    """
    entries = [PDBEntry(path) for path in pdb_paths]
    models = ""
    for i, mol in enumerate(entries):
        models += "MODEL " + str(i) + "\n"
        atoms = mol._get_atom_dicts(
            highlight_mask_dict=highlight_mask_dict,
            style=style,
            opacity=opacity,
            colors=colors,
            accent_color=accent_color,
        )
        models += "".join([str(x) for x in atoms])
        models += "ENDMDL\n"

    view = _get_view(canvas_size)
    view.addModelsAsFrames(models)

    for i, at in enumerate(atoms):
        default = {"cartoon": {"color": "black"}}
        view.setStyle({"model": -1, "serial": i + 1}, at.get("pymol", default))

    view.zoomTo()
    view.animate({"loop": direction, "step": 10})
    view.show()


def show_animation_from_pickle(
    pickle_paths,
    highlight_mask=None,
    style="cartoon",
    opacity=1,
    direction="forward",
    colors=None,
    accent_color="#D96181",
    canvas_size=(400, 300),
):
    """Show an animation of the given pickle files.

    Parameters
    ----------
    pickle_paths : list of str
        List of paths to pickle files.
    highlight_mask : numpy.ndarray, optional
        Mask to highlight. 1s indicate the atoms to highlight; assumes
        the chains to be concatenated in alphabetical order.
    style : str, optional
        The style of the visualization; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'
    opacity : float, default 1
        The opacity of the visualization.
    direction : {"forward", "backAndForth"}
        The direction of the animation.
    colors : list of str, optional
        List of colors to use for each chain
    accent_color : str, optional
        Color to use for the highlighted atoms (use `None` to disable)
    canvas_size : tuple of int, optional
        The size of the canvas to display the animation on

    """
    entries = [ProteinEntry.from_pickle(path) for path in pickle_paths]
    models = ""
    for i, mol in enumerate(entries):
        models += "MODEL " + str(i) + "\n"
        if highlight_mask is None:
            highlight_mask = mol.get_predict_mask()
        atoms = mol._get_atom_dicts(
            highlight_mask=highlight_mask,
            style=style,
            opacity=opacity,
            colors=colors,
            accent_color=accent_color,
        )
        models += "".join([str(x) for x in atoms])
        models += "ENDMDL\n"

    view = _get_view(canvas_size)
    view.addModelsAsFrames(models)

    for i, at in enumerate(atoms):
        default = {"cartoon": {"color": "black"}}
        view.setStyle({"model": -1, "serial": i + 1}, at.get("pymol", default))

    view.zoomTo()
    view.animate({"loop": direction})
    view.show()

    return view


def merge_pickle_files(paths_to_merge, save_path):
    """Merge the given pickle files into a single file.

    Parameters
    ----------
    paths_to_merge : list of str
        List of paths to pickle files to merge.
    save_path : str
        Path to save the merged file.

    """
    create_fn = ProteinEntry.from_pickle
    entries = [create_fn(path) for path in paths_to_merge]
    merged_entry = entries[0]
    for entry in entries[1:]:
        merged_entry.merge(entry)
    if save_path.endswith(".pdb"):
        merged_entry.to_pdb(save_path)
    elif save_path.endswith(".pickle"):
        merged_entry.to_pickle(save_path)
    else:
        raise ValueError("save_path must end with .pdb or .pickle")


def show_merged_pickle(
    file_paths,
    highlight_masks=None,
    style="cartoon",
    highlight_style=None,
    opacity=1.0,
    only_predicted=False,
    canvas_size=(400, 300),
):
    """Show a merged visualization of the given PDB or pickle files.

    Parameters
    ----------
    file_paths : list of str
        List of paths to PDB or pickle files.
    highlight_masks : list of numpy.ndarray, optional
        List of masks to highlight. 1s indicate the atoms to highlight; assumes
        the chains to be concatenated in alphabetical order.
    style : str, optional
        The style of the visualization; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'
    highlight_style : str, optional
        The style of the highlighted atoms; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'
        (defaults to the same as `style`)
    opacity : float or list, default 1
        The opacity of the visualization.
    only_predicted : bool, default False
        Whether to only overlay the predicted atoms.
    canvas_size : tuple of int, optional
        The size of the canvas to display the visualization on

    """
    create_fn = ProteinEntry.from_pickle
    entries = [create_fn(path) for path in file_paths]
    alphabet = list(string.ascii_uppercase)
    opacity_dict = {}
    if isinstance(opacity, float):
        opacity = [opacity] * len(entries)
    for i, entry in enumerate(entries):
        update_dict = {chain: alphabet.pop(0) for chain in entry.get_chains()}
        entry.rename_chains(update_dict)
        opacity_dict.update({chain: opacity[i] for chain in entry.get_chains()})
        if highlight_masks is not None and highlight_masks[i] is None:
            highlight_masks[i] = np.zeros(entry.get_mask().sum())
    merged_entry = entries[0]
    for entry in entries[1:]:
        if only_predicted:
            entry = entry.get_predicted_entry()
        merged_entry.merge(entry)
    if highlight_masks is not None:
        highlight_mask = np.concatenate(highlight_masks, axis=0)
    else:
        highlight_mask = None
    merged_entry.visualize(
        style=style,
        highlight_style=highlight_style,
        highlight_mask=highlight_mask,
        opacity=opacity_dict,
        canvas_size=canvas_size,
    )


def show_merged_pdb(
    file_paths,
    highlight_mask_dicts=None,
    style="cartoon",
    opacity=1.0,
    canvas_size=(400, 300),
):
    """Show a merged visualization of the given PDB or pickle files.

    Parameters
    ----------
    file_paths : list of str
        List of paths to PDB or pickle files.
    highlight_mask_dicts : list of dict, optional
        List of highlight mask dictionaries. The keys are the names of the chains, and the values are `numpy` arrays of
        1s and 0s, where 1s indicate the atoms to highlight.
    style : str, optional
        The style of the visualization; one of 'cartoon', 'sphere', 'stick', 'line', 'cross'
    opacity : float or list, default 1
        The opacity of the visualization.
    canvas_size : tuple of int, optional
        The size of the canvas to display the visualization on

    """
    create_fn = PDBEntry
    entries = [create_fn(path) for path in file_paths]
    alphabet = list(string.ascii_uppercase)
    highlight_mask_dict = {} if highlight_mask_dicts is not None else None
    opacity_dict = {}
    if isinstance(opacity, float):
        opacity = [opacity] * len(entries)
    for i, entry in enumerate(entries):
        update_dict = {chain: alphabet.pop(0) for chain in entry.get_chains()}
        entry.rename_chains(update_dict)
        opacity_dict.update({chain: opacity[i] for chain in entry.get_chains()})
        if highlight_mask_dicts is not None:
            highlight_mask_dict.update(
                {
                    update_dict[chain]: mask
                    for chain, mask in highlight_mask_dicts[i].items()
                }
            )
    merged_entry = entries[0]
    for entry in entries[1:]:
        merged_entry.merge(entry)
    merged_entry.visualize(
        style=style,
        highlight_mask_dict=highlight_mask_dict,
        opacity=opacity_dict,
        canvas_size=canvas_size,
    )
