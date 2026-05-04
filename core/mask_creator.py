import cv2
from plantcv import plantcv as pcv
import os

from core.file_manager import FileManager


class MaskCreator:
    def __init__(self, state):
        """
        Encapsulate root mask creation logic using plantcv + cv2.
        Masks are saved into sub folders within state.output_dir
        """
        self.state = state
        self.files = FileManager(state)

    def _resolve_outdir(self, mask_type: str) -> str:
        """ Helper to resolve the output directory for root masks and l-type masks. Falls back to source image dir if
            no output_dir is set.

            Args:
                mask_type (string)
                    String indicating the type of the mask to be resolved ("Root", "LT")

            Return:
                Output sub folder path depending on mask_type.
        """
        if self.state.output_dir:
            # ensure tree exists
            self.files.ensure_output_dirs()
            if mask_type == "Root":
                return self.files.subdir("rootmask")
            if mask_type == "LT":
                return self.files.subdir("ltmask")
        # Fallback: same folder as image
        return os.path.dirname(self.state.filename)

    def create_mask(self, diameter_gauss: int, threshold: int, mask_type: str) -> str:
        """ Creates a binary mask of the image stored under state.filename by simple greyscale thresholding.
            Saves that mask to the respective sub folder within state.output_dir as indicated by mask_type.
            Automatically updates the state.rootmask_path and state.ltmask_path variables.

            Args:
                diameter_gauss (int)
                    Diameter of the gaussian filter applied to the greyscale image prior to thresholding
                    This filter is needed to remove noise along the main root axis resulting from debris in the image.
                    Dropping this filter results in imprecise estimation of lateral counts. Too high values will lead
                    to removal of fine lateral roots from the masked image.

                threshold (int)
                    Greyscale threshold (0 - 255) used to create the binary root mask.

                mask_type (string)
                    String indicating the type of the mask to be generated ("Root", "LT")

            Return:
                path (string)
                    Path to the generated root mask image.
                """
        if not self.state.filename:
            raise RuntimeError("No image file loaded.")

        img = cv2.imread(self.state.filename)
        gray = pcv.rgb2gray(img)
        gray = cv2.blur(gray, (diameter_gauss, diameter_gauss))
        mask = pcv.threshold.binary(gray, threshold=threshold, object_type="dark")
        filled = pcv.fill(mask, 7000) # Fill small debris
        inverted = cv2.bitwise_not(filled)

        base = os.path.splitext(os.path.basename(self.state.filename))[0]
        outdir = self._resolve_outdir(mask_type)

        if mask_type == "Root":
            path = os.path.join(outdir, f"{base}_rootmask.tif")
            cv2.imwrite(path, inverted)
            self.state.rootmask_path = path
        elif mask_type == "LT":
            path = os.path.join(outdir, f"{base}_LTmask.tif")
            cv2.imwrite(path, inverted)
            self.state.ltmask_path = path
        else:
            raise ValueError("mask_type must be 'Root', or 'LT'.")
        return path

    def create_all(self):
        """ Helper to generate root mask and L-type mask in a single function."""
        t = self.state.thresholds
        self.create_mask(t["root_mask"], "Root")
        self.create_mask(t["lt_mask"], "LT")