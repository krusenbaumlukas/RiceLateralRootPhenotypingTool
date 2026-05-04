"""
Custom ImageCanvas widget: encapsulates zoom/pan/rotate and drawing.
Modifications based on https://github.com/ImagingSolution/PythonImageViewer/
"""

import tkinter as tk
from PIL import Image, ImageTk, ImageStat
import numpy as np
import cv2

class ImageCanvas(tk.Canvas):

    def __init__(self, master, width=500, height=650, **kwargs):
        """ Create a canvas widget that displays a PIL image with support for
                    zooming, panning and coordinate transforms based on a 3x3 affine
                    transformation matrix.

                    The affine matrix is stored in self.mat_affine and maps from image
                    coordinates to canvas coordinates. User interactions (mouse drag,
                    scroll wheel, double-click) update this matrix and then redraw the
                    transformed image.

                    Args:
                        master (tk.Widget)
                            Parent widget that owns this canvas.

                        width (int)
                            Initial width of the canvas in pixels.

                        height (int)
                            Initial height of the canvas in pixels.

                        **kwargs (dict)
                            Additional keyword arguments forwarded to tk.Canvas.
        """
        super().__init__(master, width=width, height=height, bg='white', highlightbackground='black', **kwargs)
        self.pil_image = None
        self.image_tk = None
        self.mat_affine = np.eye(3)  # 3x3 affine matrix
        self._last = None

        # Bindings
        self.bind('<Button-1>', self._on_press)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<Double-Button-1>', self._on_double)
        self.bind('<MouseWheel>', self._on_wheel)

    def set_image(self, path, first_open = False):
        """ Set the PIL image that is displayed in the canvas and reset the
            affine transform so the image fits into the available canvas area.

            Args:
                pil_image (PIL.Image.Image)
                    PIL image that should be shown in this canvas.
        """
        if path is None:
            return

        im = cv2.imread(path)
        self.pil_image = Image.fromarray(im)
        # Only zoom fit the first time an image is opened
        if first_open == True:
            self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw()

    def to_image_point(self, x, y):
        """ Map canvas coords (x,y) to image coordinates according to current affine.

            Args:
                x, y (float or int)
                    X, Y coordinates in canvas coordinate system.

            Returns:
                (tuple(float, float)) or None
                    (ix, iy) in image coordinates, as floating point values.
        """
        if self.pil_image is None:
            return None
        mat_inv = np.linalg.inv(self.mat_affine)
        ix, iy, _ = np.dot(mat_inv, (x, y, 1.0))
        return (ix, iy)

    # Interaction handlers -----------------------------------------------
    def _on_press(self, event):
        """ Mouse callback for a left button press. Stores the event position.

            Args:
                event (tk.Event)
                    tkinter event object with attributes .x and .y in canvas coordinates.
        """
        self._last = (event.x, event.y)

    def _on_drag(self, event):
        """ Mouse callback for dragging with the left button held down. The drag distance between the current mouse
            position and the last stored position is computed and used to translate the affine matrix.

            Args:
                event (tk.Event)
                    Tkinter event object with attributes .x and .y in canvas
                    coordinates.
        """
        if self._last is None:
            return
        dx = event.x - self._last[0]
        dy = event.y - self._last[1]
        self.translate(dx, dy)
        self._last = (event.x, event.y)
        self.redraw()

    def _on_double(self, event):
        """ Mouse callback for a double left-click. The image is fit into the canvas using the current canvas size.
            This is a shortcut for re-centering and re-scaling the image.
        """
        if self.pil_image is None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw()

    def _on_wheel(self, event):
        """ Mouse callback for scroll-wheel events. Scrolling up or down zooms the view in or out around the mouse
            cursor position.

            Args:
                event (tk.Event)
                    Tkinter event object. Uses attributes:
                    .x, .y   → mouse position in canvas coordinates
                    .delta   → scroll delta provided by Tk (platform specific)
        """
        if self.pil_image is None:
            return
        if event.delta < 0:
            self.scale_at(1.25, event.x, event.y)
        else:
            self.scale_at(0.8, event.x, event.y)
        self.redraw()

    def reset_transform(self):
        """ Reset the affine transform to the identity matrix."""
        self.mat_affine = np.eye(3)

    def translate(self, tx, ty):
        """ Apply a translation to the current affine transform.This composes a translation matrix T with the existing
            affine matrix: mat_affine ← T · mat_affine

            Args:
                tx, ty (float)
                    Translation in x,y-direction in canvas coordinates.
        """
        mat = np.eye(3)
        mat[0,2] = tx
        mat[1,2] = ty
        self.mat_affine = mat @ self.mat_affine

    def scale(self, s):
        """ Apply a uniform scaling to the current affine transform. This composes a scaling matrix S with the existing
            affine matrix: mat_affine ← S · mat_affine.

            Args:
                s (float)
                    Scale factor. Values > 1 zoom in, values < 1 zoom out.
        """
        mat = np.eye(3)
        mat[0,0] = s
        mat[1,1] = s
        self.mat_affine = mat @ self.mat_affine

    def scale_at(self, scale, cx, cy):
        """ Apply a uniform scaling about a specific canvas point.

            Args:
                scale (float)
                    Scale factor. Values > 1 zoom in, values < 1 zoom out.

                cx, cy (float or int)
                    X, Y coordinates in canvas coordinates around which to scale.
        """
        self.translate(-cx, -cy)
        self.scale(scale)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        """ Compute an affine transform that fits the entire image within the canvas while preserving aspect ratio,
            then apply it.

            Args:
                image_width (int)
                    Width of the image in pixels.

                image_height (int)
                    Height of the image in pixels.

        """
        canvas_w = self.winfo_width() or int(self['width'])
        canvas_h = self.winfo_height() or int(self['height'])
        if image_width == 0 or image_height == 0:
            return
        self.reset_transform()
        if (canvas_w * image_height) > (image_width * canvas_h):
            s = canvas_h / image_height
            ox = (canvas_w - image_width * s) / 2
            oy = 0
        else:
            s = canvas_w / image_width
            ox = 0
            oy = (canvas_h - image_height * s) / 2
        self.scale(s)
        self.translate(ox, oy)

    def redraw(self):
        """ Redraw the current image according to the affine transform. The method clears all existing items on the
            canvas, computes the inverse of the affine matrix, and uses PIL.Image.transform to warp the original PIL
            image into the canvas size using that inverse transform. The result is drawn at the canvas origin.

            Raises:
                ValueError
                    If self.mat_affine is not invertible.

                RuntimeError
                    If the underlying PIL image cannot be transformed for any
                    reason (e.g. corrupted image). In typical use this should
                    not occur.
        """
        self.delete('all')
        if self.pil_image is None:
            return
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        mat_inv = np.linalg.inv(self.mat_affine)
        affine_inv = (mat_inv[0,0], mat_inv[0,1], mat_inv[0,2],
                      mat_inv[1,0], mat_inv[1,1], mat_inv[1,2])
        dst = self.pil_image.transform((canvas_w, canvas_h), Image.AFFINE, affine_inv, Image.NEAREST)
        self.image_tk = ImageTk.PhotoImage(dst)
        item = self.create_image(0, 0, anchor='nw', image=self.image_tk)