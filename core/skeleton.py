"""
Skeletonization and tip/branch finding using plantcv.
Saves skeletons into ROOTSKELETONS/ and updates state paths.

Creates:
- state.root_skel_path
- state.lt_skel_path

Also populates:
- For ROOT skeleton:
    state.tips       (list[(x,y)])
    state.branches   (list[(x,y)])
- For LT skeleton:
    state.lt_tips    (list[(x,y)])
    state.lt_branches(list[(x,y)])
"""
import os
import cv2
import numpy as np
from plantcv import plantcv as pcv

from core.file_manager import FileManager


class Skeletonizer:
    def __init__(self, state):
        self.state = state
        self.files = FileManager(state)

    def _skeletonize(self, mask_path: str, suffix: str) -> str:
        if not mask_path:
            raise RuntimeError("Mask path is missing for skeletonization.")
        img = cv2.imread(mask_path)
        gray = pcv.rgb2gray(img)
        gray = cv2.bitwise_not(gray)
        skel = pcv.morphology.skeletonize(gray)

        outdir = self.files.subdir("skeletons")
        base = os.path.splitext(os.path.basename(mask_path))[0]
        outname = f"{base}_skeleton_{suffix}.tif"
        outpath = os.path.join(outdir, outname)
        cv2.imwrite(outpath, skel)
        return outpath

    def create_root(self) -> str:
        if not self.state.rootmask_path:
            raise RuntimeError("Root mask not found. Create it first.")
        path = self._skeletonize(self.state.rootmask_path, "Root")
        self.state.root_skel_path = path
        self.state.root_skel_path_base = path[:-4] + "_base.tif"

        skel_img = cv2.imread(path)
        skel_gray = pcv.rgb2gray(skel_img)
        tips_img = pcv.morphology.find_tips(skel_gray)
        branches_img = pcv.morphology.find_branch_pts(skel_gray)

        tips_coords = np.argwhere(tips_img >= 255)
        branch_coords = np.argwhere(branches_img >= 255)
        self.state.tips = [(c, r) for (r, c) in tips_coords]
        self.state.branches = [(c, r) for (r, c) in branch_coords]
        # Draw tips and branch points on skeleton
        for (x, y) in self.state.branches:
            cv2.circle(skel_img, (int(x), int(y)), 4, (255, 0, 0), 1)  # BGR: cyan
        for (x, y) in self.state.tips:
            cv2.circle(skel_img, (int(x), int(y)), 4, (0, 255, 0), 1)  # BGR: magenta
        cv2.imwrite(path, skel_img)
        print("DEBUG: self.state.root_skel_path_base")
        cv2.imwrite(self.state.root_skel_path_base,skel_img)
        return path

    def create_lt(self) -> str:
        if not self.state.ltmask_path:
            raise RuntimeError("LT mask not found. Create it first.")
        path = self._skeletonize(self.state.ltmask_path, "LT")
        self.state.lt_skel_path = path

        # Also collect LT tips/branches for LD/LT workflows
        skel_img = cv2.imread(path)
        skel_gray = pcv.rgb2gray(skel_img)
        branches_img = pcv.morphology.find_branch_pts(skel_gray)
        branch_coords = np.argwhere(branches_img >= 255)
        self.state.lt_branches = [(c, r) for (r, c) in branch_coords]

        for (x, y) in self.state.MRL_path[::max(1, 2)]: # Draw every 2nd point (sufficient at large enough diameter)
            cv2.circle(skel_img, (int(x), int(y)), radius = 1, color=(255, 0, 0), thickness=-1)
        cv2.imwrite(self.state.lt_skel_path, skel_img)

        # Trip detection dropped as it is unnecessary for any downstream analysis
        # tips_img = pcv.morphology.find_tips(skel_gray)
        # tips_coords = np.argwhere(tips_img >= 255)
        # self.state.lt_tips = [(c, r) for (r, c) in tips_coords]

        return path

    def create_all(self):
        out = {}
        if self.state.rootmask_path:
            out["root"] = self.create_root()
        if self.state.ltmask_path:
            out["lt"] = self.create_lt()
        if not out:
            raise RuntimeError("No masks available. Create masks before skeletons.")
        return out