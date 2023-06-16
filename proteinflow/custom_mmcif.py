from copy import deepcopy

from biopandas.mmcif import PandasMmcif


class CustomMmcif(PandasMmcif):
    """
    A modification of `PandasMmcif`

    Adds a `get_model` method and renames the columns to match PDB format.
    """

    def read_mmcif(self, path: str):
        x = super().read_mmcif(path)
        x.df["ATOM"].rename(
            {
                "label_comp_id": "residue_name",
                "label_seq_id": "residue_number",
                "label_atom_id": "atom_name",
                "group_PDB": "record_name",
                "Cartn_x": "x_coord",
                "Cartn_y": "y_coord",
                "Cartn_z": "z_coord",
            },
            axis=1,
            inplace=True,
        )
        x.df["ATOM"]["chain_id"] = x.df["ATOM"]["auth_asym_id"]
        return x

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
            df.df["ATOM"] = df.df["ATOM"].loc[
                df.df["ATOM"]["pdbx_PDB_model_num"] == model_index
            ]
        return df
