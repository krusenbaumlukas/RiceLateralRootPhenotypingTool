"""
Results saving utilities.
Writes the table (Geno, Main, Length, STL, LD, LT) to CSV.
"""
from __future__ import annotations
from typing import Iterable, List, Tuple
from dataclasses import dataclass
import csv
import os
from datetime import datetime

HEADERS = ["Image", "Main", "Length", "STL", "LN", "LT"]

@dataclass
class ResultsSaver:
    state: object  # AppState

    def _datafiles_dir(self) -> str:
        """ Returns the output directory. Prefer <output_dir>/DATAFILES; else fall back to image folder."""
        if getattr(self.state, "output_dir", None):
            base = os.path.join(self.state.output_dir, "DATAFILES")
            os.makedirs(base, exist_ok=True)
            return base
        # fallback: alongside the source image
        if getattr(self.state, "filename", None):
            return os.path.dirname(self.state.filename)
        # last resort: current working directory
        return os.getcwd()

    def default_path(self) -> str:
        """ Suggests a default filename based on current image + timestamp.

            Return:
                A path (self.datafiles_dir) + filename (string)
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "results"
        if getattr(self.state, "filename", None):
            base = os.path.splitext(os.path.basename(self.state.filename))[0] + "_results"
        return os.path.join(self._datafiles_dir(), f"{base}_{ts}.csv")

    def save_rows(self, path: str, rows: Iterable[Tuple[str, str, str, str, str, str]]):
        """ Write rows to CSV with headers.

            Args:
                path (string)
                    The path to which results are to be saved as .csv.

                rows (string)
                    List of tuples containing results from rice root phenotyping.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            for row in rows:
                writer.writerow(list(row))

    def extract_rows_from_tree(self, tree) -> List[Tuple[str, str, str, str, str, str]]:
        """ Extracts rows from results table (ttk.Treeview) and write to list.

            Args:
                tree (ttk.Treeview)
                    A Treeview table with 6 columns

            Return:
                rows (List)
                    A list of tuples containing the results from rice root phenotyping.
        """
        rows = []
        for item_id in tree.get_children():
            vals = tuple(tree.item(item_id, "values"))
            # Ensure 6 columns (pad/truncate defensively)
            vals = (list(vals) + ["", "", "", "", "", ""])[:6]
            rows.append(tuple(vals))
        return rows