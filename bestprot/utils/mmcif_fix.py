from biopandas.mmcif import PandasMmcif
from copy import deepcopy
import numpy as np
import pandas as pd

class CustomMmcif(PandasMmcif):
    def amino3to1(self):
        df = super().amino3to1()
        df.columns = ["chain_id", "residue_name"]
        return df

    def get_model(self, model_index: int):
        """Returns a new PandasPDB object with the dataframes subset to the given model index.
        Parameters
        ----------
        model_index : int
            An integer representing the model index to subset to.
        Returns
        ---------
        pandas_pdb.PandasPdb : A new PandasPdb object containing the
          structure subsetted to the given model.
        """

        df = deepcopy(self)

        if "ATOM" in df.df.keys():
            df.df["ATOM"] = df.df["ATOM"].loc[df.df["ATOM"]["pdbx_PDB_model_num"] == model_index]
        # if "HETATM" in df.df.keys():
        #     df.df["HETATM"] = df.df["HETATM"].loc[
        #         df.df["HETATM"]["pdbx_PDB_model_num"] == model_index
        #     ]
        # if "ANISOU" in df.df.keys():
        #     df.df["ANISOU"] = df.df["ANISOU"].loc[
        #         df.df["ANISOU"]["pdbx_PDB_model_num"] == model_index
        #     ]
        return df