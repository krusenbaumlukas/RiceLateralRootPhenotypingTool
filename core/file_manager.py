"""
File and directory helpers.
"""
import os

class FileManager:
    def __init__(self, state):
        self.state = state

    def ensure_output_dirs(self):
        if not self.state.output_dir:
            raise ValueError("Output dir not set")
        base = self.state.output_dir
        # Create canonical structure
        for sub in ["ROOTMASKS", "LTMASKS", "ROOTSKELETONS", "DATAFILES"]:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    def subdir(self, key: str) -> str:
        """Return absolute path to a known subdir (creates if missing)."""
        mapping = {
            "rootmask": "ROOTMASKS",
            "ltmask": "LTMASKS",
            "skeletons": "ROOTSKELETONS",
            "data": "DATAFILES",
        }
        if not self.state.output_dir:
            raise ValueError("Output dir not set")
        sub = mapping[key]
        path = os.path.join(self.state.output_dir, sub)
        os.makedirs(path, exist_ok=True)
        return path