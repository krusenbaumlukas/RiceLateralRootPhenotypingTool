"""
Entry point for the Root Analyzer app.
"""
import json
import os
import tkinter as tk

from ui.main_window import MainWindow
from models.state import AppState

def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def main():
    cfg = load_config()
    state = AppState.from_config(cfg)

    # instantiate and run the main window
    app = MainWindow(state)
    app.mainloop()

if __name__ == "__main__":
    main()