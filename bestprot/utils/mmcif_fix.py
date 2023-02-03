from biopandas.mmcif import PandasMmcif
from copy import deepcopy
import numpy as np
import pandas as pd

class CustomMmcif(PandasMmcif):
    def amino3to1(self):
        df = super().amino3to1()
        df.columns = ["chain_id", "residue_name"]
        return df

    def get_model_start_end(self) -> pd.DataFrame:
        """Get the start and end of the models contained in the PDB file.
        Extracts model start and end line indexes based
          on lines labelled 'OTHERS' during parsing.
        Returns
        ---------
        pandas.DataFrame : Pandas DataFrame object containing
          the start and end line indexes of the models.
        """

        other_records = self.df["OTHERS"]

        idxs = other_records.loc[other_records["record_name"] == "MODEL"]
        ends = other_records.loc[other_records["record_name"] == "ENDMDL"]
        idxs.columns = ["record_name", "model_idx", "start_idx"]
        idxs["end_idx"] = ends.line_idx.values
        # If structure only contains 1 model, create a dummy df mapping all lines to model_idx 1
        if len(idxs) == 0:
            n_lines = len(self.pdb_text.splitlines())
            idxs = pd.DataFrame(
                [
                    {
                        "record_name": "MODEL",
                        "model_idx": 1,
                        "start_idx": 0,
                        "end_idx": n_lines,
                    }
                ]
            )

        return idxs

    def label_models(self):
        """Adds a column (`"model_id"`) to the underlying
        DataFrames containing the model number."""
        idxs = self.get_model_start_end()
        # Label ATOMS
        if "ATOM" in self.df.keys():
            pdb_df = self.df["ATOM"]
            idx_map = np.piecewise(
                np.zeros(len(pdb_df)),
                [
                    (pdb_df.line_idx.values >= start_idx)
                    & (pdb_df.line_idx.values <= end_idx)
                    for start_idx, end_idx in zip(
                        idxs.start_idx.values, idxs.end_idx.values
                    )
                ],
                idxs.model_idx,
            )
            self.df["ATOM"]["model_id"] = idx_map.astype(int)
        # LABEL HETATMS
        if "HETATM" in self.df.keys():
            pdb_df = self.df["HETATM"]
            idx_map = np.piecewise(
                np.zeros(len(pdb_df)),
                [
                    (pdb_df.line_idx.values >= start_idx)
                    & (pdb_df.line_idx.values <= end_idx)
                    for start_idx, end_idx in zip(
                        idxs.start_idx.values, idxs.end_idx.values
                    )
                ],
                idxs.model_idx,
            )
            self.df["HETATM"]["model_id"] = idx_map.astype(int)
        if "ANISOU" in self.df.keys():
            pdb_df = self.df["ANISOU"]
            idx_map = np.piecewise(
                np.zeros(len(pdb_df)),
                [
                    (pdb_df.line_idx.values >= start_idx)
                    & (pdb_df.line_idx.values <= end_idx)
                    for start_idx, end_idx in zip(
                        idxs.start_idx.values, idxs.end_idx.values
                    )
                ],
                idxs.model_idx,
            )
            self.df["ANISOU"]["model_id"] = idx_map.astype(int)
        return self

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
        df.label_models()

        if "ATOM" in df.df.keys():
            df.df["ATOM"] = df.df["ATOM"].loc[df.df["ATOM"]["model_id"] == model_index]
        if "HETATM" in df.df.keys():
            df.df["HETATM"] = df.df["HETATM"].loc[
                df.df["HETATM"]["model_id"] == model_index
            ]
        if "ANISOU" in df.df.keys():
            df.df["ANISOU"] = df.df["ANISOU"].loc[
                df.df["ANISOU"]["model_id"] == model_index
            ]
        return df