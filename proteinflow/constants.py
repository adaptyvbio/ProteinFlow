"""Constants used throughout the proteinflow package."""

from collections import defaultdict, namedtuple

import numpy as np

###################################################################################
############################# Residue Constants ###################################
###################################################################################


MAIN_ATOMS = {
    "GLY": None,
    "ALA": 0,
    "VAL": 0,
    "LEU": 1,
    "ILE": 1,
    "MET": 2,
    "PRO": 1,
    "TRP": 5,
    "PHE": 6,
    "TYR": 7,
    "CYS": 1,
    "SER": 1,
    "THR": 1,
    "ASN": 1,
    "GLN": 2,
    "HIS": 2,
    "LYS": 3,
    "ARG": 4,
    "ASP": 1,
    "GLU": 2,
}

D3TO1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
}

REVERSE_D3TO1 = {v: k for k, v in D3TO1.items()}
REVERSE_D3TO1["X"] = "GLY"

ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"

FEATURES_DICT = defaultdict(lambda: defaultdict(lambda: 0))
FEATURES_DICT["hydropathy"].update(
    {
        "-": 0,
        "I": 4.5,
        "V": 4.2,
        "L": 3.8,
        "F": 2.8,
        "C": 2.5,
        "M": 1.9,
        "A": 1.8,
        "W": -0.9,
        "G": -0.4,
        "T": -0.7,
        "S": -0.8,
        "Y": -1.3,
        "P": -1.6,
        "H": -3.2,
        "N": -3.5,
        "D": -3.5,
        "Q": -3.5,
        "E": -3.5,
        "K": -3.9,
        "R": -4.5,
    }
)
FEATURES_DICT["volume"].update(
    {
        "-": 0,
        "G": 60.1,
        "A": 88.6,
        "S": 89.0,
        "C": 108.5,
        "D": 111.1,
        "P": 112.7,
        "N": 114.1,
        "T": 116.1,
        "E": 138.4,
        "V": 140.0,
        "Q": 143.8,
        "H": 153.2,
        "M": 162.9,
        "I": 166.7,
        "L": 166.7,
        "K": 168.6,
        "R": 173.4,
        "F": 189.9,
        "Y": 193.6,
        "W": 227.8,
    }
)
FEATURES_DICT["charge"].update(
    {
        **{"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1},
        **{x: 0 for x in "ABCFGIJLMNOPQSTUVWXYZ-"},
    }
)
FEATURES_DICT["polarity"].update(
    {**{x: 1 for x in "RNDQEHKSTY"}, **{x: 0 for x in "ACGILMFPWV-"}}
)
FEATURES_DICT["acceptor"].update(
    {**{x: 1 for x in "DENQHSTY"}, **{x: 0 for x in "RKWACGILMFPV-"}}
)
FEATURES_DICT["donor"].update(
    {**{x: 1 for x in "RKWNQHSTY"}, **{x: 0 for x in "DEACGILMFPV-"}}
)
CDR = {"-": 0, "H1": 1, "H2": 2, "H3": 3, "L1": 4, "L2": 5, "L3": 6}

ALLOWED_AG_TYPES = {
    "protein",
    "protein | protein",
    "protein | protein | protein",
    "protein | protein | protein | protein | protein",
    "protein | protein | protein | protein",
    np.nan,
}

SIDECHAIN_ORDER = {
    "CYS": ["CB", "SG"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "SER": ["CB", "OG"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "PRO": ["CB", "CG", "CD"],
    "THR": ["CB", "OG1", "CG2"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "GLY": [],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "ALA": ["CB"],
    "VAL": ["CB", "CG1", "CG2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "MET": ["CB", "CG", "SD", "CE"],
}

BACKBONE_ORDER = ["N", "C", "CA", "O"]
CDR_ENDS = {
    "L": {"L1": (26, 32), "L2": (50, 52), "L3": (91, 96)},
    "H": {"H1": (26, 32), "H2": (52, 56), "H3": (96, 101)},
}
CDR_VALUES = {"L": defaultdict(lambda: "-"), "H": defaultdict(lambda: "-")}
for chain_type in ["L", "H"]:
    for key, (start, end) in CDR_ENDS[chain_type].items():
        for x in range(start, end + 1):
            CDR_VALUES[chain_type][x] = key


###################################################################################
################################## S3 Constants ###################################
###################################################################################

S3Obj = namedtuple("S3Obj", ["key", "mtime", "size", "ETag"])


###################################################################################
################################# PDB Constants ###################################
###################################################################################
ALPHABET_PDB = "XACDEFGHIKLMNPQRSTVWY"

GLOBAL_PAD_CHAR = 0
ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP",
}
ATOM_MAP_4 = {a: ["N", "C", "CA", "O"] for a in ONE_TO_THREE_LETTER_MAP.keys()}
ATOM_MAP_1 = {a: ["CA"] for a in ONE_TO_THREE_LETTER_MAP.keys()}
ATOM_MAP_3 = {a: ["N", "C", "CA"] for a in ONE_TO_THREE_LETTER_MAP.keys()}


def _PMAP(x):
    return [
        FEATURES_DICT["hydropathy"][x] / 5,
        FEATURES_DICT["volume"][x] / 200,
        FEATURES_DICT["charge"][x],
        FEATURES_DICT["polarity"][x],
        FEATURES_DICT["acceptor"][x],
        FEATURES_DICT["donor"][x],
    ]
