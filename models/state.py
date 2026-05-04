import os

class AppState:
    """ This class contains all global state variables utilized during measurements of rice root phenotypic traits."""
    def __init__(self):
        # Paths and filenames
        self.filename = None            # path to the original image file (string)
        self.output_dir = None          # path to the selected output directory (string)

        # Mask / skeleton paths
        self.rootmask_path = None       # path to the root mask image file (string)
        self.rootmask_path_base = None  # path to the unannotated root mask image file (string)
        self.ltmask_path = None         # path to the L-type mask image file (string)
        self.root_skel_path = None      # path to the root skeleton image file (string)
        self.lt_skel_path = None        # path to the L-type skeleton image file (string)

        # Thresholds and settings
        self.thresholds = {"root_mask": 250, "lt_mask": 210}    # Thresholds employed during mask creation (greyscale [0-255])
        self.diam_gaus = 4                                      # Diamater of the gaussian filter applied during mask creation (px)
        self.margin_laterals = 2                                # Margin for counting of branch points to estimate lateral
                                                                # root number (both on primary and L-type) (px)
        self.margin_LT = 10                                     # Margin for counting of branch points to estimate L-type
                                                                # root number on primary (px)
        self.dpi = 600                                          # DPI of the images

        self.tips = None                 # Array of all tip points identified in the root skeleton
        self.branches = None             # Array of all branch points identified in the root skeleton
        self.lt_branches = None          # Array of all branch points identified in the L-type skeleton

        self.MRL_seed = None        # (x, y) seed center
        self.MRL_tip = None         # (x, y) main root axis tip
        self.MRL_path = None        # path of the main root axis list[(x,y)]
        self.MRL_waypoints = []     # list of intermediate (x, y) waypoints
        self.current_main_idx = 1   # Index of the current main root

        self.ll_current_tip = None  # (x,y) S-type lateral tip
        self.ll_waypoints = []      # list[(x,y)] intermediate waypoints for S-type path
        self.lt_current_tip = None  # (x,y) L-type lateral tip
        self.lt_waypoints = []      # list[(x,y)] intermediate waypoints for L-type path

        self.results = []       # Results (rows for the results table)

    @classmethod
    def from_config(cls, cfg: dict):
        """ Loads global configuration for:
                self.output_dir
                self.dpi
                self.thresholds
                self.diam_gaus
                self.margin_laterals
                self.margin_LT
            from config.json to the global state.

            Args:
                cfg (dictonary)
                    Global configuration dictionary
            """
        s = cls()
        if cfg.get("output_dir"):
            s.output_dir = cfg["output_dir"]
            os.makedirs(s.output_dir, exist_ok=True)
        if cfg.get("dpi"):
            s.dpi = cfg["dpi"]
        if cfg.get("thresholds"):
            s.thresholds.update(cfg["thresholds"])
        if "diam_gaus" in cfg:
            s.diam_gaus = cfg["diam_gaus"]
        if "margin_laterals" in cfg:
            s.margin_laterals = cfg["margin_laterals"]
        if "margin_LT" in cfg:
            s.margin_LT = cfg["margin_LT"]
        return s