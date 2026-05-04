from typing import List, Tuple, Set
import numpy as np
import cv2

Point = Tuple[int, int]

class Measurements:
    """ This class contains several helper function for measurements of root phenotypic traits in rice root scans."""
    def __init__(self, state):
        self.state = state

    # --- basic ---
    def px_to_mm(self, length_px: int, digits = 0) -> float:
        """ Computes the length of a path in mm based on the global state.dpi variable.

            Args:
                length_px (int)
                Length of a path in pixels

                digits (int)
                Number of digits to round the output

            Return:
                Length of the path in mm
        """
        return round((length_px / self.state.dpi) * 25.4, digits)

    def path_length_mm(self, path_points: List[Point], digits = 0) -> float:
        if not path_points:
            return 0
        return self.px_to_mm(len(path_points), digits)

    def count_branch_hits_radius(self,
                                path_points,
                                branch_points,
                                radius: int,
                                img_shape
                                ) -> int:
        """ Count distinct branch pixels that lie within 'radius' pixels of the
            MRL path, using a single morphological dilation.

            Args:
                path_points (list[tuple(x,y)])
                    List of points along a path

                branch_points (list[tuple(x,y)])
                    List of branch points in a skeletonized image

                radius (int)
                    The margin along the path within which branch points are to be counted

                img_shape (tuple, int)
                    (rows, cols) of the skeletonized image

            Return:
                hits (int)
                Number of branch points within the selected path + defined raius
        """
        if not path_points or not branch_points:
            return 0

        rows, cols = img_shape
        # 1) Make a blank mask and draw the path on it
        path_img = np.zeros((rows, cols), dtype=np.uint8)
        for (x, y) in path_points:
            if 0 <= x < cols and 0 <= y < rows:
                path_img[y, x] = 255

        # 2) Dilate the path with a disk / circle of given radius
        ksize = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(path_img, kernel)

        # 3) Count branch points that fall inside the dilated region
        seen = set()
        hits = 0
        for (x, y) in branch_points:
            if 0 <= x < cols and 0 <= y < rows:
                if dilated[y, x] == 255 and (x, y) not in seen:
                    seen.add((x, y))
                    hits += 1
        return hits

    # --- high level ---
    def measure_MRL(self, path_points: List[Point]):
        """ Helper for main root axis length computation."""
        return self.path_length_mm(path_points)

    def measure_ll_mm(self, lateral_path: List[Point]) -> float:
        """ Helper for S-type length computation"""
        return self.path_length_mm(lateral_path, 2)

