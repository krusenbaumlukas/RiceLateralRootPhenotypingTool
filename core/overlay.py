"""
Overlay helpers:
Draw points and paths onto root skeleton images.
"""
from typing import Iterable, Tuple, List, Optional
import cv2
import numpy as np

Point = Tuple[int, int]

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """ Helper function to ensure a loaded image is BGR for further process.

        Args:
            img (np array)
            Numpy array of loaded image

        Return:
            img (np array)
            Numpy array of image ensured to be BGR
    """
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def draw_points_on_image(path_to_image: str, points: Iterable[Point], color=(0, 255, 255), radius: int = 6, thickness: int = 2):
    """ Draw points on images. Utilized for tip- and branch points, as well as seed center and the main root axis tip.

        Args:
            path_to_image (str)
                Path to the image.

            points (List[tuple(x,y)])
                Iterable list of points (x,y) to be displayed in the image.

            color (tuple, int) (R,G,B)
                Tuple indicating the color of the points in RGB format (0-255).
                
            radius (int)
                Integer indicating the radius of the points in px.
                
            thickness (int)
                Integer indicating the thickness of the line used to draw the points 
                (A value of -1 results in filled circles)
                
        Return:
            img (numpy array)
            A numpy array of the image with points drawn on it.
    """
    img = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path_to_image)
    img = _ensure_bgr(img)
    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
    return img

def draw_points_and_path(path_to_image: str, points: Iterable[Point], path_points: Optional[List[Point]] = None):
    """ Draws a path on an image with tip points highlighted. The path is drawn by drawing a point with radius 2 
        onto the image for every 3rd pixel in the path.
        
        Args: 
            path_to_image (str)
                String indicating the path to the respective image
                
            points (list[(x,y)])
                Iterable list of tip points to be displayed in the image.
                
            path_points (list[(x,y)])
                List of points along the path to be drawn onto the image.
                
        Return:
            img (numpy array)
             A numpy array of the image with points and path drawn on it.
    """
    img = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path_to_image)
    img = _ensure_bgr(img)

    # path first
    if path_points:
        for (x, y) in path_points[::3]:
            cv2.circle(img, (int(x), int(y)), 2, (255, 255, 0), -1)
    # points on top (Seed center: light blue, Root tip: magenta if two points provided)
    pts = list(points)
    for idx, (x, y) in enumerate(pts):
        col = (0, 255, 255) if idx == 0 else (255, 0, 255)
        cv2.circle(img, (int(x), int(y)), 6, col, 2)
    return img