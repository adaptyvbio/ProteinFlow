"""
This module contains a function to annotate secondary structure elements (SSEs) in a chain.

Adapted from: https://github.com/biotite-dev/biotite.
"""

import numpy as np
from biotite.structure.geometry import angle, dihedral, distance


def _annotate_sse(X):
    """Annotation of secondary structure elements (SSEs) in a chain."""
    ca_coord = X[:, 2, :]
    length = ca_coord.shape[0]

    _r_helix = (np.deg2rad(89 - 12), np.deg2rad(89 + 12))
    _a_helix = (np.deg2rad(50 - 20), np.deg2rad(50 + 20))
    _d3_helix = ((5.3 - 0.5), (5.3 + 0.5))
    _d4_helix = ((6.4 - 0.6), (6.4 + 0.6))

    _r_strand = (np.deg2rad(124 - 14), np.deg2rad(124 + 14))
    _a_strand = (np.deg2rad(-180), np.deg2rad(-125), np.deg2rad(145), np.deg2rad(180))
    _d2_strand = ((6.7 - 0.6), (6.7 + 0.6))
    _d3_strand = ((9.9 - 0.9), (9.9 + 0.9))
    _d4_strand = ((12.4 - 1.1), (12.4 + 1.1))

    d2i = np.full(length, np.nan)
    d3i = np.full(length, np.nan)
    d4i = np.full(length, np.nan)
    ri = np.full(length, np.nan)
    ai = np.full(length, np.nan)

    d2i[1 : length - 1] = distance(ca_coord[0 : length - 2], ca_coord[2:length])
    d3i[1 : length - 2] = distance(ca_coord[0 : length - 3], ca_coord[3:length])
    d4i[1 : length - 3] = distance(ca_coord[0 : length - 4], ca_coord[4:length])
    ri[1 : length - 1] = angle(
        ca_coord[0 : length - 2], ca_coord[1 : length - 1], ca_coord[2:length]
    )
    ai[1 : length - 2] = dihedral(
        ca_coord[0 : length - 3],
        ca_coord[1 : length - 2],
        ca_coord[2 : length - 1],
        ca_coord[3 : length - 0],
    )

    sse = np.full(len(ca_coord), "c", dtype="U1")
    # Annotate helices
    # Find CA that meet criteria for potential helices
    is_pot_helix = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
            d3i[i] >= _d3_helix[0]
            and d3i[i] <= _d3_helix[1]
            and d4i[i] >= _d4_helix[0]
            and d4i[i] <= _d4_helix[1]
        ) or (
            ri[i] >= _r_helix[0]
            and ri[i] <= _r_helix[1]
            and ai[i] >= _a_helix[0]
            and ai[i] <= _a_helix[1]
        ):
            is_pot_helix[i] = True
    # Real helices are 5 consecutive helix elements
    is_helix = np.zeros(len(sse), dtype=bool)
    counter = 0
    for i in range(len(sse)):
        if is_pot_helix[i]:
            counter += 1
        else:
            if counter >= 5:
                is_helix[i - counter : i] = True
            counter = 0
    # Extend the helices by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_helix[i]:
            sse[i] = "a"
            if (d3i[i - 1] >= _d3_helix[0] and d3i[i - 1] <= _d3_helix[1]) or (
                ri[i - 1] >= _r_helix[0] and ri[i - 1] <= _r_helix[1]
            ):
                sse[i - 1] = "a"
            sse[i] = "a"
            if (d3i[i + 1] >= _d3_helix[0] and d3i[i + 1] <= _d3_helix[1]) or (
                ri[i + 1] >= _r_helix[0] and ri[i + 1] <= _r_helix[1]
            ):
                sse[i + 1] = "a"
        i += 1

    # Annotate sheets
    # Find CA that meet criteria for potential strands
    is_pot_strand = np.zeros(len(sse), dtype=bool)
    for i in range(len(sse)):
        if (
            d2i[i] >= _d2_strand[0]
            and d2i[i] <= _d2_strand[1]
            and d3i[i] >= _d3_strand[0]
            and d3i[i] <= _d3_strand[1]
            and d4i[i] >= _d4_strand[0]
            and d4i[i] <= _d4_strand[1]
        ) or (
            ri[i] >= _r_strand[0]
            and ri[i] <= _r_strand[1]
            and (
                (ai[i] >= _a_strand[0] and ai[i] <= _a_strand[1])
                or (ai[i] >= _a_strand[2] and ai[i] <= _a_strand[3])
            )
        ):
            is_pot_strand[i] = True
    # Real strands are 5 consecutive strand elements,
    # or shorter fragments of at least 3 consecutive strand residues,
    # if they are in hydrogen bond proximity to 5 other residues
    is_strand = np.zeros(len(sse), dtype=bool)
    counter = 0
    contacts = 0
    for i in range(len(sse)):
        if is_pot_strand[i]:
            counter += 1
            coord = ca_coord[i]
            for strand_coord in ca_coord:
                dist = distance(coord, strand_coord)
                if dist >= 4.2 and dist <= 5.2:
                    contacts += 1
        else:
            if counter >= 4:
                is_strand[i - counter : i] = True
            elif counter == 3 and contacts >= 5:
                is_strand[i - counter : i] = True
            counter = 0
            contacts = 0
    # Extend the strands by one at each end if CA meets extension criteria
    i = 0
    while i < len(sse):
        if is_strand[i]:
            sse[i] = "b"
            if d3i[i - 1] >= _d3_strand[0] and d3i[i - 1] <= _d3_strand[1]:
                sse[i - 1] = "b"
            sse[i] = "b"
            if d3i[i + 1] >= _d3_strand[0] and d3i[i + 1] <= _d3_strand[1]:
                sse[i + 1] = "b"
        i += 1
    return sse
