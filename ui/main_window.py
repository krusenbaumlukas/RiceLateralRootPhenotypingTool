import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import cv2
from plantcv import plantcv as pcv

from ui.image_canvas import ImageCanvas
from core.mask_creator import MaskCreator
from core.skeleton import Skeletonizer
from core.pathfinding import PathFinder
from core.measurements import Measurements
from core.file_manager import FileManager
from core.overlay import draw_points_and_path
from core.save_results import ResultsSaver


# ---- UI constants  ----
BTN_HEIGHT = 2       # all buttons
BTN_IPADY = 4        # internal vertical padding

class MainWindow(tk.Tk):
    """ This class contains the main application window."""
    def __init__(self, state):
        """ Builds the main applications window and pulls all helper functions from ./core.

            Args:
                state (AppState)
                    Contains all global state variables.
        """
        super().__init__()
        self.state = state
        self.state.active_mode = None
        pcv.params.debug = None
        pcv.params.debug_outdir = ""

        self.title("Rice lateral root phenotyping tool")
        self.geometry("975x752")
        self.resizable(True, True)
        ttk.Style().theme_use("clam")

        # master layout
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=1, sticky="ns", padx=5, pady=5)

        self._build_left_panel()
        self._build_right_panel()

        # Core helpers
        self.masker = MaskCreator(self.state)
        self.skel = Skeletonizer(self.state)
        self.pathfinder = PathFinder(self.state)
        self.measure = Measurements(self.state)
        self.files = FileManager(self.state)
        self.saver = ResultsSaver(self.state)

        self._build_menu()
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        # Bind "Done" to enter key
        self.bind('<Return>', self.enter_mode)

    # ---------------- LEFT PANEL ----------------
    def _build_left_panel(self):
        """ This functions builds the left panel of the GUI with the canvas for image display, as well as buttons
        to switch between display modes and a label indicating the position of the cursor on the displayed image."""
        lf = self.left_frame
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)  # canvas
        lf.rowconfigure(1, weight=0)  # coords
        lf.rowconfigure(2, weight=0)  # display buttons

        # Canvas
        self.canvas = ImageCanvas(lf, width=500, height=650)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Coordinate label
        self.label_image_pixel = tk.Label(lf, text="(--, --)")
        self.label_image_pixel.grid(row=1, column=0, sticky="w", pady=(4, 6))

        # Display controls (left panel)
        # Display controls (tall)
        disp = tk.LabelFrame(lf, text="Display")
        disp.grid(row=2, column=0, sticky="ew", padx=0, pady=(0, 4))
        for c in (0, 1, 2, 3):
            disp.columnconfigure(c, weight=1)

        self.btn_show_org = tk.Button(disp, text="Original", command=self.show_original, height=BTN_HEIGHT)
        self.btn_show_rootmask = tk.Button(disp, text="Root Mask", command=self.show_rootmask, height=BTN_HEIGHT)
        self.btn_show_ltmask = tk.Button(disp, text="LT Mask", command=self.show_ltmask, height=BTN_HEIGHT)
        self.btn_show_rootskel = tk.Button(disp, text="Root Skel", command=self.show_rootskel, height=BTN_HEIGHT)

        self.btn_show_org.grid(row=0, column=0, sticky="ew", padx=4, pady=4, ipady=4)
        self.btn_show_rootmask.grid(row=0, column=1, sticky="ew", padx=4, pady=4, ipady=4)
        self.btn_show_ltmask.grid(row=0, column=2, sticky="ew", padx=4, pady=4, ipady=4)
        self.btn_show_rootskel.grid(row=0, column=3, sticky="ew", padx=4, pady=4, ipady=4)

    # ---------------- RIGHT PANEL ----------------
    def _build_right_panel(self):
        """ This functions builds the right panel of the GUI including:
        1. Instruction text to guide user through the measurement process.
        2. All buttons for the mask creation logic as well as tick boxes to toggle saving of intermediate image files.
        3. All buttons for measurements.
        4. An overview table for the measured data with an option to modify data in the table on double click.
        5. A button for saving all data.
        """
        rf = self.right_frame
        for i in range(4):
            rf.columnconfigure(i, weight=1)
        for r in range(30):
            rf.rowconfigure(r, weight=0)

        # Instruction text
        self.instruction_text = tk.Text(rf, width=55, height=9, state="normal")
        self.instruction_text.grid(row=0, column=0, columnspan=4, pady=(5, 8))
        self._set_instructions(
            "1. File → Set Output Directory\n"
            "2. File → Open Image\n"
            "3. Click 'Create All Masks and Skeletons' (or create individually)\n"
        )

        # Create section
        self.create_frame = tk.LabelFrame(rf, text="Mask Creation")
        self.create_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky="ew")
        cf = self.create_frame
        cf.columnconfigure((0, 1, 2, 3), weight=1)

        # Row 1: Root Mask, LT Mask, Root Skeleton
        self.rootmask_btn = tk.Button(cf, text="Create Root Mask", command=self.create_root_mask, height=BTN_HEIGHT)
        self.ltmask_btn = tk.Button(cf, text="Create LT Mask", command=self.create_lt_mask, height=BTN_HEIGHT)
        self.root_skel_btn = tk.Button(cf, text="Create Skeleton", command=self.create_root_skeleton, height=BTN_HEIGHT)
        self.rootmask_btn.grid(row=1, column=0, sticky="ew", padx=2, pady=3, ipady=BTN_IPADY)
        self.ltmask_btn.grid(row=1, column=1, sticky="ew", padx=2, pady=3, ipady=BTN_IPADY)
        self.root_skel_btn.grid(row=1, column=2, sticky="ew", padx=2, pady=3, ipady=BTN_IPADY)

        # Row 2: Combined create all
        self.all_ms_btn = tk.Button(cf, text="Create All Masks and Skeletons",
                                    command=self.create_all_masks_and_skeletons,
                                    height=BTN_HEIGHT)
        self.all_ms_btn.grid(row=2, column=0, columnspan=3, sticky="ew", padx=2, pady=(3, 10), ipady=BTN_IPADY)

        # Column 4: File save options:
        self.save_label = tk.Label(cf, text="Keep image files:")
        self.save_label.grid(row=1, column=3, sticky="nw", pady=5, padx = 10)
        self.keep_rm = tk.IntVar()
        self.keep_ltm = tk.IntVar()
        self.keep_rs = tk.IntVar(value = 1)
        self.keep_rootmask_check = tk.Checkbutton(cf, text="Root Mask", var = self.keep_rm)
        self.keep_ltmask_check = tk.Checkbutton(cf, text="LT Mask", var=self.keep_ltm)
        self.keep_rootskeleton_check = tk.Checkbutton(cf, text="Root Skeleton", var=self.keep_rs)
        self.keep_rootmask_check.grid(row=1, column=3, sticky="sw", pady=0, padx=10)
        self.keep_ltmask_check.grid(row=2, column=3, sticky="nw", pady=0, padx=10)
        self.keep_rootskeleton_check.grid(row=2, column=3, sticky="sw", pady=12, padx=10)

        # Measurements
        self.measure_frame = tk.LabelFrame(rf, text="Measurements")
        self.measure_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky="ew")
        mf = self.measure_frame
        mf.columnconfigure((0, 1, 2, 3), weight=1)

        self.MRL_btn = tk.Button(mf, text="Measure MRL", command=self.start_MRL_workflow, height=BTN_HEIGHT)
        self.ll_btn = tk.Button(mf, text="Measure STL", command=self.start_stl_workflow, height=BTN_HEIGHT)
        self.ln_btn = tk.Button(mf, text="Measure LN", command=self.start_ln_workflow, height=BTN_HEIGHT)
        self.lt_btn = tk.Button(mf, text="Measure LT", command=self.start_lt_workflow, height=BTN_HEIGHT)
        self.reset_btn = tk.Button(mf, text="Reset", command=self.reset_mode, height=BTN_HEIGHT)
        self.done_btn = tk.Button(mf, text="Done", command=self.done_mode, height=BTN_HEIGHT)
        self.next_main_btn = tk.Button(mf, text="Next Root", command=self.next_main_root, height=BTN_HEIGHT)

        self.MRL_btn.grid(row=0, column=0, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)
        self.ll_btn.grid(row=0, column=1, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)
        self.ln_btn.grid(row=0, column=2, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)
        self.lt_btn.grid(row=0, column=3, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)
        self.reset_btn.grid(row=1, column=0, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)
        self.done_btn.grid(row=1, column=1, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)
        self.next_main_btn.grid(row=1, column=3, sticky="ew", padx=2, pady=2, ipady=BTN_IPADY)

        # Results table
        self.results_table = ttk.Treeview(
            rf, columns=("Image", "Main", "Length", "STL", "LN", "LT"),
            show="headings", height=9
        )
        for col, w in [("Image", 60), ("Main", 60), ("Length", 70),
                       ("STL", 70), ("LN", 100), ("LT", 100)]:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=w)

        # Enable in-place editing on double click or F2
        self.results_table.bind("<Double-1>", self._on_tree_double_click)
        self.results_table.bind("<F2>", self._on_tree_f2)

        # Keep a reference to any active editor widget
        self._tree_editor = None
        self._tree_edit_info = None  # (item_id, col_id)

        self.results_table.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(0, 6))

        # ---- Save and Clear Last Root buttons (aligned near bottom) ----
        self.save_btn = tk.Button(rf, text="Save Results", command=self.save_results, height=BTN_HEIGHT)
        self.clear_last_root_btn = tk.Button(
            rf, text="Clear Last Root", command=self.clear_last_root, height=BTN_HEIGHT
        )
        self.save_btn.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=4, ipady=4)
        self.clear_last_root_btn.grid(row=5, column=2, columnspan=2, sticky="ew", padx=4, pady=4, ipady=4)

    # ---------------- MENU ----------------
    def _build_menu(self):
        """ This function build the main menu. Options from the menu are:
        1. Set an output directory for all generated datafiles
            The output directory will automatically include sub folders:
            ./ROOTMASK (storing all rootmasks)
            ./LTMASK (storing all L-type masks)
            ./SEKELTONS (storing all skeletons with annotations)
            ./DATAFILES (default saving of export data)
            All image files will be deleted upon opening of a new image or saving data, unless the respective tick box
            is toggled.
        2. Open a new image
        3. Settings (Modify global state variables)
        4. Exit (Close GUI)
            """
        menu = tk.Menu(self)

        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Set Output Directory", command=self.set_output_dir)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Settings", command=self.edit_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        menu.add_cascade(label="File", menu=file_menu)
        self.config(menu=menu)

    # ---------------- UTIL ----------------
    def _set_instructions(self, text: str):
        """ This function writes instructions to the instructions text panel.
        Args:
            text (string)
                An instruction text to be displayed.
        """
        self.instruction_text.config(state="normal")
        self.instruction_text.delete("1.0", "end")
        self.instruction_text.insert("end", text)
        self.instruction_text.config(state="disabled")

    def _unbind_rclick(self):
        """ Removes functionality of right mouse click used for selection of points in the image."""
        self.canvas.unbind("<Button-3>")

    def _round_point(self, ip_float):
        """ Rounds floating point coordinates of clicks to integer.
            Args:
                ip_float (tuple)
                    tuple of floating point numbers (x y coordinates)
            Returns:
                tuple of integer x y coordinates
        """
        if not ip_float:
            return None
        return (int(round(ip_float[0])), int(round(ip_float[1])))

    def _load_skeleton_mask(self, path):
        """ Reads an RGB image and returns as greyscale
        Args:
            path (str)
                Path to the image file
        Returns:
            gray (np array)
                A numpy array of the greyscale image
        """
        img = cv2.imread(path)
        gray = pcv.rgb2gray(img)
        return gray

    def _current_row_id(self):
        """ Helper function to select the number of rows already in the table"""
        items = self.results_table.get_children()
        return items[-1] if items else None

    def _show_root_skeleton(self):
        """ Displays root skeleton from global state variable state.root_skel_path (str)"""
        if getattr(self.state, "root_skel_path", None):
            self.canvas.set_image(self.state.root_skel_path)

    def _show_root_skeleton_base(self):
        """
        Show the baseline root skeleton image with tips/branches,
        without any MRL/LL/LT overlays.
        """
        if self.state.root_skel_path_base:
            self.canvas.set_image(self.state.root_skel_path_base)
        else:
            self.canvas.set_image(self.state.root_skel_path)

    def _on_tree_f2(self, event):
        """ Enables editing of the results table when pressing F2."""
        # F2 edits the first selected cell (defaults to first column if header area)
        sel = self.results_table.selection()
        if not sel:
            return
        item_id = sel[0]
        # If mouse is over a column, use it; else default to first column
        x, y = self.results_table.winfo_pointerx() - self.results_table.winfo_rootx(), \
               self.results_table.winfo_pointery() - self.results_table.winfo_rooty()
        col = self.results_table.identify_column(x) or "#1"
        self._start_edit_cell(item_id, col)

    def _on_tree_double_click(self, event):
        """ Enables editing of the results table on double-click of a cell."""
        # Identify the row/column under the cursor
        region = self.results_table.identify("region", event.x, event.y)
        if region != "cell":
            return
        item_id = self.results_table.identify_row(event.y)
        col = self.results_table.identify_column(event.x)  # "#1", "#2", ...
        if not item_id or not col:
            return
        self._start_edit_cell(item_id, col)

    def _start_edit_cell(self, item_id, col):
        """ Editing of the results table.
        Args:
            item_id (int)
                row of the selected cell

            col (int)
                column of the selected cell
        """
        # Cancel an existing editor
        self._cancel_tree_editor()

        # Compute the cell bbox to place our Entry on top
        bbox = self.results_table.bbox(item_id, col)
        if not bbox:
            return  # not visible
        x, y, w, h = bbox

        # Map Treeview column id to logical column name
        # Treeview columns are: ("Image", "Main", "Length", "STL", "LN", "LT")
        col_index = int(col.replace("#", "")) - 1
        columns = ["Image", "Main", "Length", "STL", "LN", "LT"]
        if not (0 <= col_index < len(columns)):
            return
        column_name = columns[col_index]

        # Current value
        current = self.results_table.set(item_id, column_name)

        # Create editor Entry
        editor = tk.Entry(self.results_table)
        editor.insert(0, current)
        editor.select_range(0, tk.END)
        editor.focus_set()

        # Place it over the cell
        editor.place(x=x, y=y, width=w, height=h)

        # Wire up commit/cancel
        editor.bind("<Return>", lambda e: self._commit_tree_editor())
        editor.bind("<KP_Enter>", lambda e: self._commit_tree_editor())
        editor.bind("<Escape>", lambda e: self._cancel_tree_editor())
        editor.bind("<FocusOut>", lambda e: self._commit_tree_editor())  # commit on focus out

        self._tree_editor = editor
        self._tree_edit_info = (item_id, column_name)

    def _commit_tree_editor(self):
        """ Commit function to save changes to the results table"""
        if not self._tree_editor or not self._tree_edit_info:
            return
        new_val = self._tree_editor.get()
        item_id, column_name = self._tree_edit_info

        # Commit to the Treeview
        self.results_table.set(item_id, column_name, new_val)

        # Cleanup
        self._tree_editor.destroy()
        self._tree_editor = None
        self._tree_edit_info = None

    def _cancel_tree_editor(self):
        """ Cancel editing of the results table. Toggled on pressing ESC"""
        if self._tree_editor:
            try:
                self._tree_editor.destroy()
            except Exception:
                pass
        self._tree_editor = None
        self._tree_edit_info = None

    # ---------------- CANVAS EVENTS ----------------
    def _on_canvas_motion(self, event):
        """ Function to display the position of the cursor in the image.
        Args:
            event (tuple, int)
                tuple indicating the cursor position on the canvas.
        """
        ip = self.canvas.to_image_point(event.x, event.y)
        if ip:
            self.label_image_pixel.config(text=f"({ip[0]:.1f}, {ip[1]:.1f})")
        else:
            self.label_image_pixel.config(text="(--, --)")

    # ---------------- FILE/IMAGE ----------------
    def set_output_dir(self):
        """ Toplevel to select the output directory."""
        out = filedialog.askdirectory()
        if not out:
            return
        self.state.output_dir = out
        try:
            self.files.ensure_output_dirs()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._set_instructions(f"Output directory set:\n{out}\nOpen an image and create masks/skeletons.")

    def open_image(self):
        """ Toplevel to select the next image to be analyzed.
        Resets all analysis variables from previous image."""
        fname = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.bmp *.tif")],
            parent = self
        )
        if not fname:
            return
        self.state.filename = fname
        self.canvas.set_image(fname, first_open = True)
        base = os.path.basename(fname)

        self._set_instructions(
            f"Loaded: {base}\n→ Click 'Create All Masks and Skeletons' or create individually."
        )

        # Delete all intermediate files of previous analysis as indicated
        if self.keep_rm.get() == 0 and not self.state.rootmask_path == None:
            os.remove(self.state.rootmask_path)
        if self.keep_ltm.get() == 0 and not self.state.ltmask_path == None:
            os.remove(self.state.ltmask_path)
        if self.keep_rs.get() == 0 and not self.state.rootmask_path == None:
            os.remove(self.state.root_skel_path)
        # Reset all analysis variables and states
        # Mask / skeleton paths
        self.state.rootmask_path = None
        self.state.ltmask_path = None
        self.state.root_skel_path = None
        self.state.root_skel_path_base = None
        self.state.lt_skel_path = None

        # Analysis arrays
        self.state.tips = None
        self.state.branches = None

        # MRL workflow selections & results
        self.state.MRL_seed = None  # (x, y) seed center
        self.state.MRL_tip = None  # (x, y) main root axis tip
        self.state.MRL_path = None  # list[(x,y)]
        self.state.current_main_idx = 1

    def edit_settings(self):
        """
        Modal dialog to edit Root/LT mask thresholds in self.state.thresholds.
        """
        top = tk.Toplevel(self)
        top.title("Settings")
        top.transient(self)
        top.grab_set()  # modal
        top.resizable(False, False)

        # Layout
        frm = ttk.Frame(top, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        # Current values (default to sane ints if missing)
        root_val = tk.IntVar(value=int(self.state.thresholds.get("root_mask", 250)))
        lt_val = tk.IntVar(value=int(self.state.thresholds.get("lt_mask", 200)))
        diam_gauss = tk.IntVar(value=int(self.state.diam_gaus))
        margin_laterals = tk.IntVar(value=int(self.state.margin_laterals))
        margin_LT = tk.IntVar(value=int(self.state.margin_LT))
        current_dpi = tk.IntVar(value=int(self.state.dpi))

        # Rootmask + LT mask thresholds
        # Labels + spinboxes (0–255 range)
        ttk.Label(frm, text="Root mask threshold (0–255):").grid(row=0, column=0, sticky="w", pady=(0, 6))
        sp_root = tk.Spinbox(frm, from_=0, to=255, textvariable=root_val, width=6, justify="right")
        sp_root.grid(row=0, column=1, sticky="w", pady=(0, 6))

        ttk.Label(frm, text="LT mask threshold (0–255):").grid(row=1, column=0, sticky="w", pady=(0, 6))
        sp_lt = tk.Spinbox(frm, from_=0, to=255, textvariable=lt_val, width=6, justify="right")
        sp_lt.grid(row=1, column=1, sticky="w", pady=(0, 6))

        # Gaussian blur
        # Labels + spinboxes (0–100 range)
        ttk.Label(frm, text="Diameter Gaussian blur (px):").grid(row=3, column=0, sticky="w", pady=(0, 6))
        sp_gausblur = tk.Spinbox(frm, from_=0, to=100, textvariable=diam_gauss, width=6, justify="right")
        sp_gausblur.grid(row=3, column=1, sticky="w", pady=(0, 6))

        # Margin for lateral branch counting
        # Labels + spinboxes (0–100 range)
        ttk.Label(frm, text="Margin for lateral branch counting (px):").grid(row=4, column=0, sticky="w", pady=(0, 6))
        sp_lateralmargin = tk.Spinbox(frm, from_=0, to=100, textvariable=margin_laterals, width=6, justify="right")
        sp_lateralmargin.grid(row=4, column=1, sticky="w", pady=(0, 6))

        # Margin for L-type branch counting
        # Labels + spinboxes (0–100 range)
        ttk.Label(frm, text="Margin for L-type branch counting (px):").grid(row=5, column=0, sticky="w", pady=(0, 6))
        sp_LTmargin = tk.Spinbox(frm, from_=0, to=100, textvariable=margin_LT, width=6, justify="right")
        sp_LTmargin.grid(row=5, column=1, sticky="w", pady=(0, 6))

        # DPI
        # Labels + Combobox
        ttk.Label(frm, text="Image DPI:").grid(row=6, column=0, sticky="w", pady=(0, 6))
        DPI_combobox = ttk.Combobox(frm, width=5, textvariable=current_dpi)
        DPI_combobox['values'] = ('400',
                                  '600',
                                  '800',
                                  '1600')
        DPI_combobox.grid(column=1, row=6, sticky="e", pady=(0, 6))

        # Buttons
        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=2, sticky="e", pady=(8, 0))

        def on_cancel():
            top.destroy()

        def on_save():
            """ Saves all updated variables to self.state.

                Raises:
                    ValueError
                        When inputs are in unreasonable ranges
                        (< 0 or > 255 for root_val and lt_val)
                        ()< 0 or > 100 for diam_guass, margin_laterals, margin_LT)
            """
            try:
                rv = int(root_val.get())
                lv = int(lt_val.get())
                dg= int(diam_gauss.get())
                ml = int(margin_laterals.get())
                mLT = int(margin_LT.get())
                dpi = int(current_dpi.get())
                if not (0 <= rv <= 255 and
                        0 <= lv <= 255 and
                        0 <= dg <= 100 and
                        0 <= ml <= 100 and
                        0 <= mLT <= 100):
                    raise ValueError
            except Exception:
                messagebox.showerror("Invalid input", "Please enter integers between 0 and 255.")
                return
            # Persist to state
            self.state.thresholds["root_mask"] = rv
            self.state.thresholds["lt_mask"] = lv
            self.state.diam_gaus = dg
            self.state.margin_laterals = ml
            self.state.margin_LT = mLT
            self.state.dpi = dpi
            top.destroy()

        ttk.Button(btns, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text="Save", command=on_save).grid(row=0, column=1)

        # Center over parent
        self.update_idletasks()
        px = self.winfo_rootx() + (self.winfo_width() // 2)
        py = self.winfo_rooty() + (self.winfo_height() // 2)
        top.update_idletasks()
        tw, th = top.winfo_width(), top.winfo_height()
        top.geometry(f"+{max(0, px - tw // 2)}+{max(0, py - th // 2)}")

        # Block until closed
        self.wait_window(top)

    # ---------------- CREATE (buttons) ----------------
    def create_root_mask(self):
        """ Creates the root mask of the current image based on the global state root mask threshold and the global
            state diameter for the gaussian filter during mask creation.

            Raises:
                Error
                    When no image is opened.
                Error
                    For any Toplevel exception.

        """
        if not self.state.filename:
            messagebox.showerror("Error", "Open an image first.")
            return
        try:
            path = self.masker.create_mask(self.state.diam_gaus, self.state.thresholds["root_mask"], "Root")
            self.canvas.set_image(path)
            self._set_instructions("Root mask created. You can create LT mask and Root skeleton, or use the combined button.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_lt_mask(self):
        """ Creates the L-type mask of the current image based on the global state L-type mask threshold and the global
            state diameter for the gaussian filter during mask creation.

            Raises:
                Error
                    When no image is opened.
                Error
                    For any Toplevel exception.
        """
        if not self.state.filename:
            messagebox.showerror("Error", "Open an image first.")
            return
        try:
            path = self.masker.create_mask(self.state.diam_gaus, self.state.thresholds["lt_mask"], "LT")
            self.canvas.set_image(path)
            self._set_instructions("LT mask created.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_root_skeleton(self):
        """ Creates the root skeleton from the generated root mask.

            Raises:
                Error
                    When no root mask is yet created.
                Error
                    For any Toplevel exception.
        """
        if not self.state.rootmask_path:
            messagebox.showerror("Error", "Create the Root mask first.")
            return
        try:
            out = self.skel.create_root()
            self.canvas.set_image(out)
            self._set_instructions("Root skeleton created.\nTips and branches extracted for analysis.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def create_all_masks_and_skeletons(self):
        """ Combines:
            - Create Root mask
            - Create LT mask
            - Create Root skeleton

        Raises:
            Error
                When no image is opened.
            Error
                For any Toplevel exception.

        Note:
            LT skeletons are only created upon measurements of lateral numbers
        """
        if not self.state.filename:
            messagebox.showerror("Error", "Open an image first.")
            return
        try:
            # masks
            self.masker.create_mask(self.state.diam_gaus, self.state.thresholds["root_mask"], "Root")
            self.masker.create_mask(self.state.diam_gaus, self.state.thresholds["lt_mask"], "LT")
            # skeletons
            self.skel.create_root()
            # show root skeleton by default
            if self.state.root_skel_path:
                self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions("All masks & skeletons created (Root + LT). You can proceed to measurements.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------- DISPLAY ----------------
    def show_original(self):
        """ Displays the original image in the canvas.

            Raises:
                Error
                    When no image file is selected.
        """
        if not self.state.filename:
            messagebox.showerror("Error", "No file selected.")
            return
        self.canvas.set_image(self.state.filename)

    def show_rootmask(self):
        """ Displays the root mask in the canvas.

            Raises:
                Error
                    When no root mask is created.
        """
        if not self.state.rootmask_path:
            messagebox.showerror("Error", "No root mask created.")
            return
        self.canvas.set_image(self.state.rootmask_path)

    def show_ltmask(self):
        """ Displays the L-type mask in the canvas.

            Raises:
                Error
                    When no L-type mask is created.
        """
        if not self.state.ltmask_path:
            messagebox.showerror("Error", "No L-type mask created.")
            return
        self.canvas.set_image(self.state.ltmask_path)

    def show_rootskel(self):
        """ Displays the root skeleton in the canvas. Tips and branch-points, as well as paths generated during the
            analysis are drawn into the skeleton image and automatically displayed.

            Raises:
                Error
                    When no skeleton is created.
        """
        if not getattr(self.state, "root_skel_path", None):
            messagebox.showerror("Error", "No root skeleton created.")
            return
        self.canvas.set_image(self.state.root_skel_path)

    # ---------------- MEASUREMENTS: MRL ----------------
    def start_MRL_workflow(self):
        """ Starts the measurement of the length of the main root axis by binding functionality of selection for the
            seed, the tip of the main root axis and intermediate waypoints to the right mouse click.

            Raises:
                Error
                    When no skeleton is created.
        """
        if not self.state.root_skel_path:
            messagebox.showerror("Error", "Create the Root skeleton first.")
            return
        self.state.active_mode = "MRL"
        self.state.MRL_seed = None
        self.state.MRL_tip = None
        self.state.MRL_path = None
        self.state.MRL_waypoints = []
        self._set_instructions(
            "MRL mode:\n"
            "1) Right-click Seed Center (SC)\n"
            "2) Right-click Main Root Axis Tip\n"
            "3) (Optional) Right-click extra waypoints along the desired main root\n"
            "4) Click 'Done' to compute MRL. (Or hit Enter)"
        )
        self.canvas.bind("<Button-3>", self._on_MRL_click)

    def _on_MRL_click(self, event):
        """ Selection of seed, main root axis tip and additional (optional) waypoints based on right mouse click.
            Automatically rounds selected point to integer values and snaps to the closest non-black pixel in the
            root skeleton.

            Args:
                event (tuple, float)
                    Tuple of a selected pixel in image

            Raises:
                Error
                    When no pixel in skeleton == 255
                Error
                    When no path is found by A*

            Note:
                Draws the path to the root skeleton. Displays the root skeleton and computes the length of the main root
                axis.
        """
        ip = self.canvas.to_image_point(event.x, event.y)
        pt = self._round_point(ip)
        if not pt:
            return

        # snap to nearest non blck pixel in skeleton
        skel_mask = self._load_skeleton_mask(self.state.root_skel_path)
        ys, xs = np.where(skel_mask == 255)
        if len(xs) == 0:
            messagebox.showerror("Error", "Skeleton has no walkable pixels.")
            return
        d2 = (xs - pt[0])**2 + (ys - pt[1])**2
        idx = int(np.argmin(d2))
        snapped = (int(xs[idx]), int(ys[idx]))

        # 1st click -> Seed Center
        if self.state.MRL_seed is None:
            self.state.MRL_seed = snapped
            disp = draw_points_and_path(self.state.root_skel_path, [self.state.MRL_seed], None)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions("MRL mode:\nSeed Center set.\nRight-click the main root axis tip.")
            return

        # 2nd click -> Main root axis tip
        if self.state.MRL_tip is None:
            self.state.MRL_tip = snapped
            pts = [self.state.MRL_seed, self.state.MRL_tip]
            disp = draw_points_and_path(self.state.root_skel_path, pts, None)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions(
                "MRL mode:\n"
                "Main root axis tip set.\n"
                "Optionally right-click extra waypoints along the desired main root.\n"
                "Then click 'Done' to compute MRL. (Or hit Enter)"
            )
            return

        # 3rd+ clicks → waypoints
        self.state.MRL_waypoints.append(snapped)
        # all points in order: Seed, Tip, Waypoints
        pts = [self.state.MRL_seed, self.state.MRL_tip] + self.state.MRL_waypoints
        disp = draw_points_and_path(self.state.root_skel_path, pts, None)
        cv2.imwrite(self.state.root_skel_path, disp)
        self.canvas.set_image(self.state.root_skel_path)
        self._set_instructions(
            "MRL mode:\n"
            f"{len(self.state.MRL_waypoints)} waypoint(s) set.\n"
            "Add more waypoints or click 'Done' to compute MRL. (Or hit Enter)"
        )

    def _finish_MRL_path_with_waypoints(self):
        """
        Build the path for the main root axis using A* with intermediate waypoints.

        Path sequence (in image coords):
            Tip -> waypoint_1 -> waypoint_2 -> ... -> waypoint_n -> Seed
        Each segment is found by a separate A* call on the root skeleton mask.
        """
        if not self.state.MRL_seed or not self.state.MRL_tip:
            messagebox.showerror("Error", "MRL mode: Seed and MRAT must be selected before finishing.")
            return

        skel_mask = self._load_skeleton_mask(self.state.root_skel_path)

        # build ordered list of waypoints: Tip -> ...waypoints... -> Seed
        points = [self.state.MRL_tip] + list(self.state.MRL_waypoints) + [self.state.MRL_seed]

        full_path = []
        try:
            for i in range(len(points) - 1):
                start = points[i]
                goal = points[i + 1]
                seg = self.pathfinder.astar(start, goal, skel_mask)
                if not seg:
                    raise RuntimeError(f"No path found between {start} and {goal} on skeleton.")
                if full_path:
                    # avoid duplicating join node
                    full_path.extend(seg[1:])
                else:
                    full_path.extend(seg)
        except Exception as e:
            messagebox.showerror("A* Error", str(e))
            return

        self.state.MRL_path = full_path

        # draw final path + all points on skeleton
        all_pts = [self.state.MRL_seed, self.state.MRL_tip] + self.state.MRL_waypoints
        disp = draw_points_and_path(self.state.root_skel_path, all_pts, full_path)
        cv2.imwrite(self.state.root_skel_path, disp)
        self.canvas.set_image(self.state.root_skel_path)

        # compute MRL in mm and add/update results row
        MRL_mm = self.measure.measure_MRL(full_path)
        base = os.path.splitext(os.path.basename(self.state.filename))[0] if self.state.filename else "NA"

        # insert new row for this main root axis
        self.results_table.insert(
            "",
            "end",
            values=(base, str(self.state.current_main_idx), MRL_mm, "", "", "")
        )

        self._set_instructions(
            f"MRL computed: {MRL_mm} mm.\n"
            "You can now measure LL/LD/LT or click 'Next Main Root >>'."
        )

    # ---------------- MEASUREMENTS: STL ----------------
    def start_stl_workflow(self):
        """ Initiates the workflow for measurements of S-type length by binding functionality for the
            selection of S-type tips to the right mouse click.

            Raises:
                Error
                    When no skeleton or main root axis path are available
        """
        if not self.state.root_skel_path or not self.state.MRL_path:
            messagebox.showerror("Error", "Measure MRL first (and create root skeleton).")
            return
        self.state.active_mode = "stl"
        self.state.ll_current_tip = None
        self.state.ll_waypoints = []
        self._set_instructions(
            "LL mode (S-type lengths):\n"
            "Right-click S-type tips. Each click traces to MRL and appends length (mm) to STL."
        )
        self.canvas.bind("<Button-3>", self._on_stl_click)
        self._show_root_skeleton()

    def _on_stl_click(self, event):
        """ Selection a single S-type tip based on right mouse click.
            Automatically rounds selected point to integer values and snaps to the closest non-black pixel in the
            root skeleton. Optional waypoints along the S-type can be selected.

            Args:
                event (tuple, float)
                    Tuple of a selected pixel in image

            Note:
                Draws tip and optional waypoints to the root skeleton. Displays the root skeleton.

            Raises:
                Error
                    When no pixel in skeleton == 25
        """
        ip = self.canvas.to_image_point(event.x, event.y)
        pt = self._round_point(ip)
        if not pt:
            return

        skel_mask = self._load_skeleton_mask(self.state.root_skel_path)
        ys, xs = np.where(skel_mask == 255)
        if len(xs) == 0:
            messagebox.showerror("Error", "Root skeleton has no walkable pixels.")
            return

        d2 = (xs - pt[0]) ** 2 + (ys - pt[1]) ** 2
        idx = int(np.argmin(d2))
        snapped = (int(xs[idx]), int(ys[idx]))

        if self.state.ll_current_tip is None:
            # First click: tip
            self.state.ll_current_tip = snapped
            self.state.ll_waypoints = []

            disp = draw_points_and_path(self.state.root_skel_path, [snapped], None)
            cv2.imwrite(self.state.root_skel_path,disp)
            self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions(
                "STL mode:\n"
                "Tip set.\n"
                "Right-click additional waypoints OR press 'Done' (Or hit Enter) to compute lateral path."
            )
        else:
            # Subsequent clicks: waypoints
            self.state.ll_waypoints.append(snapped)
            pts = [self.state.ll_current_tip] + self.state.ll_waypoints
            disp = draw_points_and_path(self.state.root_skel_path, [snapped], None)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions(
                "STL mode:\n"
                "Waypoint added.\n"
                "Add more waypoints or press 'Done' (Or hit Enter) to compute the path."
            )

    def _compute_path_with_waypoints(self, start, waypoints, final_goal_path, skel_mask):
        """
        A* path from `start` to `final_goal_path` via intermediate waypoints.

        The path is built in segments:
            start -> wp1 -> wp2 -> ... -> wpN -> (nearest point on final_goal_path)

        Segments start->wp* use standard A*.
        Last segment uses A* to any point on `final_goal_path` (your astar_pathedgoal).

        Args:
            start (tuple (x,y)):
                Start pixel in skeleton coordinates.

            waypoints (list[tuple (x,y)]):
                Optional intermediate pixels the path must pass through, in order.

            final_goal_path (list[tuple (x,y)]):
                The MRL path (or any path) used as a pathed goal.

            skel_mask (2D numpy array):
                Skeleton image where walkable pixels == 255.

        Returns:
            path (list[tuple (x,y)]):
                Entire path from start to the goal path.

        Raises:
            RuntimeError if any segment cannot be found.
        """
        full_path = []
        current = start

        # 1) segments through explicit waypoints
        for wp in waypoints:
            seg = self.pathfinder.astar(current, wp, skel_mask)
            if not seg:
                raise RuntimeError(f"No path found between waypoint segment {current} -> {wp}")
            if full_path:
                seg = seg[1:]  # avoid duplicating joints
            full_path.extend(seg)
            current = wp

        # 2) final segment: current -> MRL path (pathed goal)
        seg_last = self.pathfinder.astar_pathedgoal(current, final_goal_path, skel_mask)
        if not seg_last:
            raise RuntimeError("No path found from last waypoint to MRL path.")

        if full_path:
            seg_last = seg_last[1:]
        full_path.extend(seg_last)

        return full_path

    # ---------------- MEASUREMENTS: LN ----------------
    def start_ln_workflow(self):
        """ Starts the workflow for getting lateral counts along the main root axis path.
            Extracted number of laterals is stored in resuts table as:
            "No. of all laterals/No. of L-types;"

            Raises:
                Error
                    When no main root axis path is available
        """
        if not self.state.MRL_path:
            messagebox.showerror("Error", "Measure MRL first.")
            return
        # Generate LT skeleton on mask with thick MRL path
        self.skel.create_lt()
        # Get total and l-type counts along pr path
        skel_mask = self._load_skeleton_mask(self.state.root_skel_path)
        rows, cols = skel_mask.shape
        # Dilating by selected margin for lateral counting
        all_count = self.measure.count_branch_hits_radius(self.state.MRL_path, self.state.branches, radius = self.state.margin_laterals,
                                                        img_shape=(rows, cols))
        #  Dilating by selected margin for L-type counting
        l_count = self.measure.count_branch_hits_radius(self.state.MRL_path, self.state.lt_branches, radius = self.state.margin_LT,
                                                        img_shape=(rows, cols))
        row_id = self._current_row_id()
        if row_id is None:
            base = os.path.splitext(os.path.basename(self.state.filename))[0] if self.state.filename else "NA"
            row_id = self.results_table.insert("", "end", values=(base, str(self.state.current_main_idx), "", "", "", ""))
            self.results_table.see(row_id)
        old = self.results_table.set(row_id, "LN")
        entry = f"{all_count}/{l_count};"
        self.results_table.set(row_id, "LN", ((old + " ") if old else "") + entry)
        self._set_instructions(f"LN appended: S={all_count}, L={l_count}.")

    # ---------------- MEASUREMENTS: LT ----------------
    def start_lt_workflow(self):
        """ Initiates the workflow for measurements of L-type length by binding functionality for the
            selection of L-type tips to the right mouse click.

            Raises:
                Error
                    When no root skeleton or main root axis path is available
        """
        if not self.state.root_skel_path or not self.state.MRL_path:
            messagebox.showerror("Error", "Measure MRL first (and create root skeleton).")
            return
        self.state.active_mode = "lt"
        self.state.lt_current_tip = None
        self.state.lt_waypoints = []
        self._set_instructions(
            "LT mode (L-type):\n"
            "Right-click L-type tips. Each click traces to MRL, computes length(mm) and #S, appending 'len/count;' to LT."
        )
        self.canvas.bind("<Button-3>", self._on_lt_click)
        self._show_root_skeleton()

    def _on_lt_click(self, event):
        """ Selection a single L-type tip based on right mouse click.
            Automatically rounds selected point to integer values and snaps to the closest non-black pixel in the
            root skeleton. Optional waypoints along the S-type can be selected.

            Args:
                event (tuple, float)
                    Tuple of a selected pixel in image

            Note:
                Draws tip and optional waypoints to the root skeleton. Displays the root skeleton.

            Raises:
                Error
                    When no pixel in skeleton == 25
        """
        ip = self.canvas.to_image_point(event.x, event.y)
        pt = self._round_point(ip)
        if not pt:
            return

        skel_mask = self._load_skeleton_mask(self.state.root_skel_path)
        ys, xs = np.where(skel_mask == 255)
        if len(xs) == 0:
            messagebox.showerror("Error", "Root skeleton has no walkable pixels.")
            return

        d2 = (xs - pt[0]) ** 2 + (ys - pt[1]) ** 2
        idx = int(np.argmin(d2))
        snapped = (int(xs[idx]), int(ys[idx]))

        if self.state.lt_current_tip is None:
            # First click: tip
            self.state.lt_current_tip = snapped
            self.state.lt_waypoints = []

            disp = draw_points_and_path(self.state.root_skel_path, [snapped], None)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions(
                "LT mode:\n"
                "Tip set.\n"
                "Right-click additional waypoints OR press 'Done' (Or hit Enter) to compute lateral path."
            )
        else:
            # Subsequent clicks: waypoints
            self.state.lt_waypoints.append(snapped)
            pts = [self.state.lt_current_tip] + self.state.lt_waypoints
            disp = draw_points_and_path(self.state.root_skel_path, [snapped], None)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)
            self._set_instructions(
                "LT mode:\n"
                "Waypoint added.\n"
                "Add more waypoints or press 'Done' (Or hit Enter) to compute the path."
            )

    # ---------------- RESET / DONE / NEXT MAIN ROOT ----------------
    def reset_mode(self):
        """ Resets all selected tips to restart the measurement workflow.

            Note:
                Previous measurements need to be manually removed from the results table.
        """

        mode = getattr(self.state, "active_mode", None)
        if mode is None:
            self._set_instructions("Nothing to reset.")
            return
        if mode == "MRL":
            self.state.MRL_seed = None
            self.state.MRL_tip = None
            self.state.MRL_path = None
            self._show_root_skeleton()
            self._set_instructions("MRL reset. Right-click the seed center, then right click the main root axis tip.")
        elif mode == "stl":
            self._show_root_skeleton()
            self.state.ll_current_tip = None
            self.state.ll_waypoints = []
            self._set_instructions("LL reset. Right-click S-type tips to measure again.")
        elif mode == "lt":
            self._show_root_skeleton()
            self.state.lt_current_tip = None
            self.state.lt_waypoints = []
            self._set_instructions("LT reset. Right-click L-type tips to measure again.")

    def enter_mode(self,event):
        """
        Helper function to trigger done_mode upon enter key.
        """
        self.done_mode()

    def done_mode(self):
        """ Unbinds functionality of right clicks when MRL, STL or LT is selected.
            Triggers pathfinding by A* for MRL, STL, LT.
        """
        mode = getattr(self.state, "active_mode", None)

        if mode == "stl":
            # compute ST path with waypoints
            tip = self.state.ll_current_tip
            if tip is None:
                self._unbind_rclick()
                self.state.active_mode = None
                self._set_instructions("STL mode finished (no tip selected).")
                return

            try:
                skel_mask = self._load_skeleton_mask(self.state.root_skel_path)
                path = self._compute_path_with_waypoints(
                    start=tip,
                    waypoints=self.state.ll_waypoints,
                    final_goal_path=self.state.MRL_path,
                    skel_mask=skel_mask
                )
                self.state.ll_current_tip = None
                self.state.ll_waypoints = []

            except Exception as e:
                messagebox.showerror("STL A* Error", str(e))
                self._unbind_rclick()
                self.state.active_mode = None
                return

            # show final path
            pts = [tip] + self.state.ll_waypoints
            disp = draw_points_and_path(self.state.root_skel_path, pts, path)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)

            # measure and update STL column
            mm = self.measure.measure_ll_mm(path)
            row_id = self._current_row_id()
            if row_id is None:
                base = os.path.splitext(os.path.basename(self.state.filename))[0] if self.state.filename else "NA"
                row_id = self.results_table.insert("", "end",
                                                   values=(base, str(self.state.current_main_idx), "", "", "", ""))

            old = self.results_table.set(row_id, "STL")
            self.results_table.set(row_id, "STL", ((old + ";") if old else "") + f"{mm}")

            self._set_instructions(f"STL path computed. Length = {mm} mm.")
            return

        if mode == "lt":
            # compute LT path with waypoints
            tip = self.state.lt_current_tip
            if tip is None:
                self._unbind_rclick()
                self.state.active_mode = None
                self._set_instructions("LT mode finished (no tip selected).")
                return

            try:
                skel_mask = self._load_skeleton_mask(self.state.root_skel_path)
                path = self._compute_path_with_waypoints(
                    start=tip,
                    waypoints=self.state.lt_waypoints,
                    final_goal_path=self.state.MRL_path,
                    skel_mask=skel_mask
                )
                self.state.lt_current_tip = None
                self.state.lt_waypoints = []
            except Exception as e:
                messagebox.showerror("LT A* Error", str(e))
                self._unbind_rclick()
                self.state.active_mode = None
                return

            pts = [tip] + self.state.lt_waypoints
            disp = draw_points_and_path(self.state.root_skel_path, pts, path)
            cv2.imwrite(self.state.root_skel_path, disp)
            self.canvas.set_image(self.state.root_skel_path)

            length_mm = self.measure.measure_ll_mm(path)
            rows, cols, channels = disp.shape
            s_on_lat = self.measure.count_branch_hits_radius(path, self.state.branches, radius = self.state.margin_laterals,
                                                        img_shape=(rows, cols))

            row_id = self._current_row_id()
            if row_id is None:
                base = os.path.splitext(os.path.basename(self.state.filename))[0] if self.state.filename else "NA"
                row_id = self.results_table.insert("", "end",
                                                   values=(base, str(self.state.current_main_idx), "", "", "", ""))

            old = self.results_table.set(row_id, "LT")
            entry = f"{length_mm}/{s_on_lat};"
            self.results_table.set(row_id, "LT", ((old + " ") if old else "") + entry)

            self._set_instructions(f"LT path computed. Length = {length_mm} mm, S-branches = {s_on_lat}.")
            return

        # --- MRL mode ---
        self._unbind_rclick()
        self.state.active_mode = None
        if mode == "MRL":
            self._set_instructions("MRL done. You may measure LL/LD/LT or click Next Main Root >>.")
            self._finish_MRL_path_with_waypoints()

            # Draw thick path to LT mask (utilized in Lateral numbers count workflow later)
            ltmask = cv2.imread(self.state.ltmask_path)
            radius = int(round(self.state.dpi * 0.05, 0))  # Approx 2.5 mm in diameter
            for (x, y) in self.state.MRL_path[::max(1, 10)]:  # Draw every 10th point (sufficient at large enough diameter)
                cv2.circle(ltmask, (int(x), int(y)), radius=radius, color=(0, 0, 0), thickness=-1)
            cv2.imwrite(self.state.ltmask_path, ltmask)
        else:
            self._set_instructions("Idle.")

    def next_main_root(self):
        """ Switches to next main root in the same image. Resets global variables

            state.MRL_seed
            state.MRL_tip,
            state.MRL_path,

            and increases the index of the current main root (current_main_idx) to save results to a new row in the
            table.
            """
        # Clear visual + state
        base_skel = cv2.imread(self.state.root_skel_path_base)
        cv2.imwrite(self.state.root_skel_path, base_skel)
        self.show_rootskel()

        self.state.active_mode = None
        self._unbind_rclick()
        # MRL-related
        if hasattr(self.state, "MRL_seed"):
            self.state.MRL_seed = None
        if hasattr(self.state, "MRL_tip"):
            self.state.MRL_tip = None
        if hasattr(self.state, "MRL_path"):
            self.state.MRL_path = None

        # LL/LT-related tip lists (if they exist)
        if hasattr(self.state, "ll_tips"):
            self.state.ll_tips = []
        if hasattr(self.state, "lt_tips_selected"):
            self.state.lt_tips_selected = []

        # If you added any extra MRL waypoints, clear them too:
        if hasattr(self.state, "MRL_waypoints"):
            self.state.MRL_waypoints = []

        # Increment root index
        self.state.current_main_idx += 1
        base = os.path.splitext(os.path.basename(self.state.filename))[0] if self.state.filename else "NA"
        base_skel = cv2.imread(self.state.root_skel_path_base)
        cv2.imwrite(self.state.root_skel_path, base_skel)
        self.canvas.set_image(self.state.root_skel_path)
        self._show_root_skeleton()
        self._set_instructions(
            f"Main Root advanced to #{self.state.current_main_idx}.\n"
            "Measure MRL for this axis, then LL/LN/LT as needed."
        )

    # ---------------- CLEAR LAST ROOT -----------------------
    def clear_last_root(self):
        """
        1. Delete the last row in the Treeview (if any).
        2. Clear selected points and paths in state.
        3. Redraw the root skeleton baseline (with tips/branches only).
        """
        # 1) Delete last row
        items = self.results_table.get_children()
        if items:
            last_id = items[-1]
            self.results_table.delete(last_id)

        # 2) Clear selection state (MRL + laterals) and unbind right-click
        self._unbind_rclick()

        # MRL-related
        if hasattr(self.state, "MRL_seed"):
            self.state.MRL_seed = None
        if hasattr(self.state, "MRL_tip"):
            self.state.MRL_tip = None
        if hasattr(self.state, "MRL_path"):
            self.state.MRL_path = None

        # LL/LT-related tip lists (if they exist)
        if hasattr(self.state, "ll_tips"):
            self.state.ll_tips = []
        if hasattr(self.state, "lt_tips_selected"):
            self.state.lt_tips_selected = []

        # If you added any extra MRL waypoints, clear them too:
        if hasattr(self.state, "MRL_waypoints"):
            self.state.MRL_waypoints = []

        # 3) Reset root skeleton to base without paths and points
        base_skel = cv2.imread(self.state.root_skel_path_base)
        cv2.imwrite(self.state.root_skel_path, base_skel)
        self.show_rootskel()
        self._set_instructions(
            "Last root cleared.\n"
            "You can measure MRL again for this main root or move to Next Main Root."
        )

    # ---------------- SAVE RESULTS ----------------
    def save_results(self):
        """ Toplevel to save results as .csv. Opens ./DATAFILES as default.

            Raises:
                Error
                    When the results table is empty

                Error
                    When saving failed
        """
        # Delete all intermediate files of previous analysis as indicated
        if self.keep_rm.get() == 0 and not self.state.rootmask_path == None:
            os.remove(self.state.rootmask_path)
        if self.keep_ltm.get() == 0 and not self.state.ltmask_path == None:
            os.remove(self.state.ltmask_path)
        if self.keep_rs.get() == 0 and not self.state.rootmask_path == None:
            os.remove(self.state.root_skel_path)

        rows = self.saver.extract_rows_from_tree(self.results_table)
        if not rows:
            messagebox.showwarning("Nothing to save", "The results table is empty.")
            return
        default_path = self.saver.default_path()
        path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            initialfile=os.path.basename(default_path),
            initialdir=os.path.dirname(default_path),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not path:
            return
        try:
            self.saver.save_rows(path, rows)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        messagebox.showinfo("Saved", f"Results written to:\n{path}")