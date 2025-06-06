#!/usr/bin/env python3
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Canvas, Frame, Button, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageEnhance, ImageGrab, ImageFilter, ImageChops
from scipy.special import expit
import os
import math
import numpy as np
import sys
import onnxruntime as ort
from scipy.ndimage import binary_dilation, binary_erosion
from timeit import default_timer as timer
import cv2
from copy import deepcopy
import platform
from screeninfo import get_monitors
#from line_profiler import profile

DEFAULT_ZOOM_FACTOR = 1.2
if getattr(sys, 'frozen', False):
    SCRIPT_BASE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"Working directory: {SCRIPT_BASE_DIR}")
MODEL_ROOT = os.path.join(SCRIPT_BASE_DIR, "Models/")
STATUS_PROCESSING = "#f00"
STATUS_NORMAL = "#000000"
PAINT_BRUSH_DIAMETER = 18
UNDO_STEPS=20
SOFTEN_RADIUS=1
MIN_RECT_SIZE = 5

pillow_formats = [
            ("All Image Files", "*.bmp *.gif *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG *.tif *.tiff *.webp"),
            ("BMP", "*.bmp"),
            ("GIF", "*.gif"),
            ("JPEG", "*.jpg *.jpeg *.JPG *.JPEG"),
            ("PNG", "*.png *.PNG"),
            ("TIFF", "*.tif *.tiff"),
            ("WEBP", "*.webp"),
            ("All Files", "*.*")
        ]


class BackgroundRemoverGUI:
    def __init__(self, root, image_paths):
        
        self.root = root

        self.image_paths = image_paths if image_paths else []
        self.current_image_index = 0
        
        self.file_count = ""
        if len(self.image_paths) > 1:
            self.file_count = f' - Image {self.current_image_index + 1} of {len(self.image_paths)}'
        elif len(self.image_paths) == 1:
            self.file_count = f' - Image 1 of 1'

        self.root.title("Background Remover" + self.file_count)
        
        self.coordinates = []
        self.labels=[]
        self.bgcolor = None
        self.dots=[]
        
        
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.last_x, self.last_y = 0, 0
        self.pan_step = 30
        
        self.lines = []
        self.lines_id = []
        self.lines_id2 = []

        if self.image_paths:
            image_path = self.image_paths[self.current_image_index]
            print(f"Image Loaded: {image_path}")

            self.original_image = Image.open(image_path)
            self.original_image = ImageOps.exif_transpose(self.original_image)
            self.image_exif = self.original_image.info.get('exif')
            print("EXIF data found" if self.image_exif else "No EXIF data found.")
        else:
            # default blank image
            self.original_image = Image.new("RGBA",(500,500),0)

            self.image_exif = None # set now so exists when loading clipboard images
        
        self.save_file_type = "png"
        self.save_file_quality = 90
        
        self.after_id = None  # ID of the scheduled after() call to render higher quality preview if no futher zooming
        self.zoom_delay = 0.2
        
        s = ttk.Style()
        s.theme_use('alt')
        
        self.style = ttk.Style()
        self.style.configure("Processing.TButton", foreground="red")

        # previously used monitor size to calculate canvas size
        # but think i've fixed resizing now, so use dummy values

        #m = [m for m in get_monitors() if m.is_primary][0]
        self.canvas_w = 200 # (m.width -300) //2
        self.canvas_h = 200 # m.height-100 

        # This is the default tkinter initalisation size
        # If the script notices these dimensions have changed, widgets have been added to the canvas

        self.init_width = 200
        self.init_height = 200
        
        self.setup_image_display()

        # Maximising the window doesn't work until after __init__ has run
        # (although full screen would work)
        # so use the resize event to maximise the interface, and allow user resizing the window
        self.root.bind("<Configure>", self.on_resize)

        if platform.system() == "Windows":
            self.root.state('zoomed')
        elif platform.system() == "Darwin":
            self.root.state('zoomed') # untested on mac
        else:
            self.root.attributes('-zoomed', True)

        self.root.update_idletasks()
        self.build_gui()
        self.update_input_image_preview()

        self.model_output_mask = Image.new("L", (int(self.orig_image_crop.width), 
                                            int(self.orig_image_crop.height)),0)

        self.set_keybindings()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def on_resize(self, event):
       
        if event.widget==self.root:
            if not self.root.winfo_height() == self.init_height or not self.root.winfo_width() == self.init_width:
                # The currently displayed preview that is being used will change, so 
                # a new encoding has to be calculated
                if hasattr(self, "encoder_output"):
                    delattr(self, "encoder_output")
                self.init_width = self.root.winfo_width()
                self.init_height = self.root.winfo_height()
                #print(f"Window dimensions {self.root.winfo_height()} {self.root.winfo_width()}")

                # previously I was setting the canvas size based on window dimensions
                # and i think this was causing a loop
                # I think fixed it now, and can use the actual canvas dimensions that are set to auto expand

                # self.canvas_w = (self.init_width -300) //2
                # self.canvas_h = self.init_height - 40
                # self.canvas.config(width=self.canvas_w, height=self.canvas_h)
                # self.canvas2.config(width=self.canvas_w, height=self.canvas_h)
                self.root.update()
                self.canvas_w = self.canvas.winfo_width()
                self.canvas_h = self.canvas.winfo_height()

                #print(f"Input canvas size {self.canvas_w} x {self.canvas_h}")
                #print(f"Output canvas size {self.canvas2.winfo_width()} x {self.canvas2.winfo_height()}")
                # this is from setup_image_display but without clearing the working image
                self.lowest_zoom_factor = min(self.canvas_w / self.original_image.width, self.canvas_h / self.original_image.height)
                self.zoom_factor = self.lowest_zoom_factor
                self.view_x = 0
                self.view_y = 0
                self.min_zoom=True
                # make a checkerboard that is always larger than preview window
                # because both canvas might not be same size if the frame for both canvases doesnt divide by 2
                self.checkerboard = self.create_checkerboard(self.canvas_w * 2, self.canvas_h * 2, square_size = 10)

                self.update_input_image_preview(Image.BOX)

      

    
    def setup_image_display(self):

        self.lowest_zoom_factor = min(self.canvas_w / self.original_image.width, self.canvas_h / self.original_image.height)
  
        self.working_image = Image.new("RGBA", self.original_image.size, (0, 0, 0, 0))
        self.working_mask = Image.new("L", self.original_image.size, 0)
        
        self.undo_history_mask = []
        self.undo_history_mask.append(self.working_mask.copy())

        self.zoom_factor = self.lowest_zoom_factor
        self.view_x = 0
        self.view_y = 0
        self.min_zoom=True

        # make a checkerboard that is always larger than preview window
        # because both canvas might not be same size if the frame for both canvases doesnt divide by 2
        self.checkerboard = self.create_checkerboard(self.canvas_w * 2, self.canvas_h * 2, square_size = 10)
        
        
    def create_checkerboard(self,width, height, square_size):
        num_squares_x = width // square_size
        num_squares_y = height // square_size
        
        img = Image.new('RGBA', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        colors = [(128, 128, 128), (153, 153, 153)]  # 'grey' and 'grey60'
        
        for row in range(num_squares_y):
            for col in range(num_squares_x):
                color = colors[(row + col) % 2]
                top_left = (col * square_size, row * square_size)
                bottom_right = ((col + 1) * square_size, (row + 1) * square_size)
                draw.rectangle([top_left, bottom_right], fill=color)
        
        return img
    
    


    def build_gui(self):
        
        self.root.minsize(width=1024, height=700)  

        # Generated by pybubu designer. 
        self.main_frame = tk.Frame(self.root, container=False, name="main_frame")
        #self.main_frame.configure(height=200, width=200)

        self.editor_frame = ttk.Frame(self.main_frame, name="editor_frame")
        #self.editor_frame.configure(height=200, width=200)
        self.input_frame = tk.Frame(self.editor_frame, name="input_frame")
        #self.input_frame.configure(height=10, width=10)
        self.canvas_input_label = ttk.Label(
            self.input_frame, name="canvas_input_label")
        self.canvas_input_label.configure(text='Input')
        self.canvas_input_label.pack(padx=10, side="top")
        self.canvas = tk.Canvas(self.input_frame, name="canvas")
        self.canvas.pack(expand=True, fill="both", side="top")
        self.input_frame.pack(expand=True, fill="both", side="left")
        separator_1 = ttk.Separator(self.editor_frame)
        separator_1.configure(orient="vertical")
        separator_1.pack(expand=False, fill="y", side="left")
        self.output_frame = tk.Frame(self.editor_frame, name="output_frame")
        #self.output_frame.configure(height=10, width=10)
        self.output = ttk.Label(self.output_frame, name="output")
        self.output.configure(text='Output')
        self.output.pack(padx=10, side="top")
        self.canvas2 = tk.Canvas(self.output_frame, name="canvas2")
        self.canvas2.pack(expand=True, fill="both", side="top")
        self.output_frame.pack(expand=True, fill="both", side="left")
        separator_3 = ttk.Separator(self.editor_frame)
        separator_3.configure(orient="vertical")
        separator_3.pack(expand=False, fill="y", side="left")
        self.Controls = tk.Frame(self.editor_frame, name="controls")
        self.Controls.configure(height=500, width=300)
        
        self.OpenImage = tk.Frame(self.Controls, name="openimage")
        self.OpenImage.configure(borderwidth=1, relief="groove")
        
        self.open_nav_frame = ttk.Frame(self.OpenImage, name="open_nav_frame")

        self.OpnImg = ttk.Button(self.open_nav_frame, name="opnimg")
        self.OpnImg.configure(text='Open Image')
        self.OpnImg.pack(side="left", padx=1)
        self.OpnImg.configure(command=self.load_image_from_dialog) 

        self.OpnClp = ttk.Button(self.open_nav_frame, name="opnclp")
        self.OpnClp.configure(text='Open Clipboard')
        self.OpnClp.pack(side="left", padx=1)
        self.OpnClp.configure(command=self.load_clipboard)

        # --- Add Next Image Button ---
        self.NextImg = ttk.Button(self.open_nav_frame, name="nextimg")
        self.NextImg.configure(text='Next Image >')
        self.NextImg.pack(side="left", padx=1)
        self.NextImg.configure(command=self.load_next_image)
        # Disable initially if less than 2 images were passed
        if len(self.image_paths) <= 1:
            self.NextImg.configure(state=tk.DISABLED)

        self.open_nav_frame.pack(side="top")
        
        self.editimageloadmask = ttk.Frame(
            self.OpenImage, name="editimageloadmask")
        self.editimageloadmask.configure(width=200)
        

        self.EditImage = ttk.Button(self.editimageloadmask, name="editimage")
        self.EditImage.configure(text='Edit Image')
        self.EditImage.pack(side="left")
        self.EditImage.configure(command=self.edit_image)

        self.LoadMask = ttk.Button(self.editimageloadmask, name="loadmask")
        self.LoadMask.configure(text='Load Mask')
        self.LoadMask.pack(expand=True, side="left")
        self.LoadMask.configure(command=self.load_mask)

        self.editimageloadmask.pack(side="top")

        self.OpenImage.pack(fill="x", padx=2, pady=3, side="top")
        self.ModelSelection = tk.Frame(self.Controls, name="modelselection")
        self.ModelSelection.configure(
            borderwidth=1, height=200, relief="groove", width=200)
        self.model_select_label = tk.Label(
            self.ModelSelection, name="model_select_label")
        self.model_select_label.configure(text='Model Selection:')
        self.model_select_label.pack(
            anchor="center", fill="x", pady=2, side="top")
        self.SegmentAnything = tk.Frame(
            self.ModelSelection, name="segmentanything")
        self.SegmentAnything.configure(height=200, width=200)
        self.sam_label = tk.Label(self.SegmentAnything, name="sam_label")
        self.sam_label.configure(text='Segment Anything')
        self.sam_label.pack(side="left")
        self.sam_combo = ttk.Combobox(self.SegmentAnything, name="sam_combo")
        self.sam_model_choice = tk.StringVar()
        self.sam_combo.configure(
            state="readonly",
            takefocus=False,
            textvariable=self.sam_model_choice,
            values=['No Models Found'],
            width=22)
        self.sam_combo.pack(side="right")
        self.SegmentAnything.pack(fill="x", padx=2, pady=2, side="top")
        self.Whole = tk.Frame(self.ModelSelection, name="whole")
        self.Whole.configure(height=200, width=200)
        self.whole_image_label = tk.Label(self.Whole, name="whole_image_label")
        self.whole_image_label.configure(text='Whole-Image')
        self.whole_image_label.pack(side="left")
        self.whole_image_combo = ttk.Combobox(
            self.Whole, name="whole_image_combo")
        self.whole_image_combo.configure(
            state="readonly",
            takefocus=False,
            values=['No Models Found'],
            width=22)
        self.whole_image_combo.pack(side="right")
        self.Whole.pack(fill="x", padx=2, side="top")
        self.ModelSelection.pack(fill="x", padx=2, pady=3, side="top")
        self.Edit = tk.Frame(self.Controls, name="edit")
        self.Edit.configure(
            borderwidth=1,
            height=200,
            relief="groove",
            width=200)
        label_9 = tk.Label(self.Edit)
        label_9.configure(text='Edit:')
        label_9.pack(anchor="center", fill="x", pady=2, side="top")
        self.SubEdit = tk.Frame(self.Edit, name="subedit")
        self.SubEdit.configure(height=200, width=200)
        self.EditCluster = tk.Frame(self.SubEdit, name="editcluster")
        self.EditCluster.configure(height=200, width=200)
        self.Add = ttk.Button(self.EditCluster, name="add")
        self.Add.configure(text='Add mask')
        self.Add.pack(expand=True, side="left")
        self.Add.configure(command=self.add_to_working_image)
        self.Remove = ttk.Button(self.EditCluster, name="remove")
        self.Remove.configure(text='Subtract mask')
        self.Remove.pack(expand=True, side="left")
        self.Remove.configure(command=self.subtract_from_working_image)
        self.Undo = ttk.Button(self.EditCluster, name="undo")
        self.Undo.configure(text='Undo')
        self.Undo.pack(expand=True, side="left")
        self.Undo.configure(command=self.undo)
        self.EditCluster.pack(expand=True, fill="x", pady=3, side="top")
        self.SubEdit.pack(fill="x", side="top")
        self.whole_image_button = ttk.Button(
            self.Edit, name="whole_image_button")
        self.whole_image_button.configure(text='Run whole-image model')
        self.whole_image_button.pack(fill="x", padx=2, pady=2, side="top")
        self.ClrPoint = ttk.Button(self.Edit, name="clrpoint")
        self.ClrPoint.configure(text='Clear Points and Mask Overlay')
        self.ClrPoint.pack(fill="x", padx=2, pady=2, side="top")
        self.ClrPoint.configure(command=self.clear_coord_overlay)
        self.ClrArea = ttk.Button(self.Edit, name="clrarea")
        self.ClrArea.configure(text='Clear Visible Area')
        self.ClrArea.pack(fill="x", padx=2, pady=2, side="top")
        self.ClrArea.configure(command=self.clear_visible_area)
        self.RstOut = ttk.Button(self.Edit, name="rstout")
        self.RstOut.configure(text='Clear Output Image')
        self.RstOut.pack(fill="x", padx=2, pady=2, side="top")
        self.RstOut.configure(command=self.clear_working_image)
        self.RstEverything = ttk.Button(self.Edit, name="rsteverything")
        self.RstEverything.configure(text='Reset Everything!')
        self.RstEverything.pack(fill="x", padx=2, pady=2, side="top")
        self.RstEverything.configure(command=self.reset_all)
        self.copy_in_out = ttk.Button(self.Edit, name="copy_in_out")
        self.copy_in_out.configure(text='Copy Input to Output')
        self.copy_in_out.pack(fill="x", padx=2, pady=2, side="top")
        self.copy_in_out.configure(command=self.copy_entire_image)
        self.Edit.pack(fill="x", padx=2, pady=3, side="top")
        self.Options = tk.Frame(self.Controls, name="options")
        self.Options.configure(
            borderwidth=1,
            height=200,
            relief="groove",
            width=200)
        label_12 = tk.Label(self.Options)
        label_12.configure(text='Options:')
        label_12.pack(anchor="center", fill="x", pady=2, side="top")
        self.BgSel = tk.Frame(self.Options, name="bgsel")
        self.BgSel.configure(height=200, width=200)
        label_15 = tk.Label(self.BgSel)
        label_15.configure(text='Background')
        label_15.pack(side="left")
        self.bg_color = ttk.Combobox(self.BgSel, name="bg_color")
        self.bg_color.configure(
            state="readonly",
            values='Transparent White Black Red Blue Orange Yellow Green Grey Lightgrey Brown Blurred_(Slow)',
            width=16)
        self.bg_color.pack(expand=False, side="right")
        self.BgSel.pack(fill="x", padx=2, pady=2, side="top")
        
        self.paint_ppm_frame = tk.Frame(self.Options, name="paint_ppm_frame")
        self.paint_ppm_frame.configure(height=200, width=200)
        
        self.ManPaint = tk.Checkbutton(self.paint_ppm_frame, name="manpaint")
        self.paint_mode = tk.BooleanVar()
        self.ManPaint.configure(
            text='Manual Paintbrush',
            variable=self.paint_mode)
        self.ManPaint.pack(fill="x", side="left")
        self.ManPaint.configure(command=self.paint_mode_toggle)
        
        self.show_mask_checkbox = tk.Checkbutton(self.paint_ppm_frame, name="show_mask_checkbox")
        self.show_mask_var = tk.BooleanVar()
        self.show_mask_checkbox.configure(
            text='Show Mask',
            variable=self.show_mask_var,
            command=self.update_output_image_preview)
        self.show_mask_checkbox.pack(fill="x", side="left")

        
        self.paint_ppm_frame.pack(side="top")

        

        self.PostMask = tk.Checkbutton(self.Options, name="postmask")
        self.ppm_var = tk.BooleanVar()
        self.PostMask.configure(
            text='Post Process Model Mask',
            variable=self.ppm_var)
        self.PostMask.pack()

        self.soften_mask_checkbox = tk.Checkbutton(self.Options, name="soften_mask_checkbox")
        self.soften_mask_var = tk.BooleanVar()
        self.soften_mask_checkbox.configure(
            text='Soften Model Mask/Paintbrush',
            variable=self.soften_mask_var)
        self.soften_mask_checkbox.pack()

        self.enable_shadow_var = tk.BooleanVar()
        self.EnableShadow = tk.Checkbutton(self.Options, text="Enable Drop Shadow (Slow)", variable=self.enable_shadow_var)
        self.EnableShadow.pack(fill="x", side="top", pady=5)
        self.EnableShadow.configure(command=self.toggle_shadow_options)

        # Frame for shadow options (initially hidden)
        self.shadow_options_frame = tk.Frame(self.Options)
        self.shadow_options_frame.pack(fill="x", padx=5, pady=5)

        self.shadow_opacity_label = tk.Label(self.shadow_options_frame, text="Opacity:")
        self.shadow_opacity_slider = ttk.Scale(self.shadow_options_frame, from_=0, to=1, orient="horizontal")
        self.shadow_opacity_slider.set(0.5)
        #self.shadow_opacity_slider.configure(command=lambda event: self.update_output_image_preview())
        self.shadow_opacity_slider.configure(command=lambda event: self.add_drop_shadow())

        self.shadow_x_label = tk.Label(self.shadow_options_frame, text="X Offset:")
        self.shadow_x_slider = ttk.Scale(
            self.shadow_options_frame, 
            from_=-200, 
            to=200, 
            orient="horizontal",
        )
        self.shadow_x_slider.set(50)
        self.shadow_x_slider.configure(command=lambda event: self.add_drop_shadow())

        self.shadow_y_label = tk.Label(self.shadow_options_frame, text="Y Offset:")
        self.shadow_y_slider = ttk.Scale(
            self.shadow_options_frame, 
            from_=-200, 
            to=200, 
            orient="horizontal",
        )
        self.shadow_y_slider.set(50)
        self.shadow_y_slider.configure(command=lambda event: self.add_drop_shadow())

        self.shadow_radius_label = tk.Label(self.shadow_options_frame, text="Blur Radius:")
        self.shadow_radius_slider = ttk.Scale(
                    self.shadow_options_frame, 
                    from_=1, 
                    to=50, 
                    orient="horizontal",
                )
        self.shadow_radius_slider.set(10)
        self.shadow_radius_slider.configure(command=lambda event: self.add_drop_shadow())

        # Initially hide the shadow options
        #self.toggle_shadow_options()

        self.Options.pack(fill="x", padx=2, pady=3, side="top")
        self.save_png = ttk.Button(self.Controls, name="save_png")
        self.save_png.configure(text='Save Image As....')
        self.save_png.pack(fill="x", padx=2, pady=3, side="top")
        self.save_png.configure(command=self.save_as_image)
        self.save_jpeg = ttk.Button(self.Controls, name="save_jpeg")
        self.save_jpeg.configure(text='Quick Save (JPEG white background)')
        self.save_jpeg.pack(fill="x", padx=2, pady=3, side="top")
        self.save_jpeg.configure(command=self.quick_save_jpeg)
        self.HelpAbout = ttk.Button(self.Controls, name="helpabout")
        self.HelpAbout.configure(text='Help / About')
        self.HelpAbout.pack(fill="x", padx=2, pady=3, side="bottom")
        self.HelpAbout.configure(command=self.show_help)
        self.Controls.pack(fill="y", side="right")
        self.editor_frame.pack(expand=True, fill="both", side="top")
        self.messagerow = tk.Frame(self.main_frame, name="messagerow")
        self.messagerow.configure(background="#ffffff", height=30)
        self.status_label = tk.Label(self.messagerow, name="status_label")
        self.status_label.configure(justify="left", text='Status: Idle')
        self.status_label.pack(expand=True, fill="x", side="left")
        self.zoom_label = tk.Label(self.messagerow, name="zoom_label")
        self.zoom_label.configure(text='Zoom: 23%', width=20)
        self.zoom_label.pack(side="left")
        self.messagerow.pack(fill="x", side="right")
        self.main_frame.pack(expand=True, fill="both", side="top")


        self.bg_color.current(0)
        self.bg_color.bind("<<ComboboxSelected>>", self.bg_color_select)
        self.sam_combo.current(0)
        self.whole_image_combo.current(0)
        self.whole_image_button.configure(command = lambda: self.run_whole_image_model(None))

        # partial string match so will also match quantised versions e.g. rmbg2_quant_q4
        sam_models = [
            "mobile_sam",
            "sam_vit_b_01ec64",
            "sam_vit_h_4b8939",
            "sam_vit_l_0b3195",
            ]

        whole_models = [
                "rmbg1_4",
                "rmbg2",
                "isnet-general-use",
                "isnet-anime",
                "u2net",
                "u2net_human_seg",
                "BiRefNet", # all birefnet variations
        ]

        matches = []

        for partial_name in sam_models:
            for filename in os.listdir('Models/'):
                filename = filename.replace(".encoder.onnx","").replace(".decoder.onnx","")
                if partial_name in filename:
                    matches.append(filename)

        if len(matches) == 0:
            messagebox.showerror("No segment anything models found in Models folder", "Please see the readme on Github for model download links.")
        else:
            models = " ".join(list(dict.fromkeys(matches)))
            print("SAM models found:", models)
            self.sam_combo.configure(values=models)
            self.sam_combo.current(0)

        matches = []

        for partial_name in whole_models:
            for filename in os.listdir('Models/'):
                if partial_name in filename and ".onnx" in filename:
                    matches.append(filename.replace(".onnx",""))

        if len(matches) == 0:
            messagebox.showerror("No whole-image models found in Models folder","Please see the readme on Github for model download links.")
        else:
            models = " ".join(list(dict.fromkeys(matches)))
            print("Whole image models found: ", models)
            self.whole_image_combo.configure(values=models)
            self.whole_image_combo.current(0)

    def bg_color_select(self, event=None):
        if self.bg_color.get() == "Blurred_(Slow)":
            self.show_blur_options()
        else:
            self.update_output_image_preview()

    def show_blur_options(self):
        option_window = tk.Toplevel(self.root)
        option_window.title("Blur Options")
        option_window.geometry("300x100")
        option_window.resizable(False, False)
              
        option_window.transient(self.root)

        blur_radius = tk.IntVar(value=30)  # Default blur radius

        tk.Label(option_window, text="Blur Radius:").pack(anchor="w", padx=10, pady=(10, 0))

        blur_frame = ttk.Frame(option_window)
        blur_frame.pack(fill="x", padx=10)

        blur_slider = ttk.Scale(
            blur_frame,
            from_=1,
            to=100,
            orient=tk.HORIZONTAL,
            variable=blur_radius,
        )
        blur_slider.pack(side="left", fill="x", expand=True)

        blur_value_label = ttk.Label(blur_frame, text=str(blur_radius.get()), width=3)
        blur_value_label.pack(side="left", padx=(5, 0))

        def update_blur_value(event=None):
            blur_value_label.config(text=str(blur_radius.get()))

        blur_slider.configure(command=update_blur_value)

        def on_ok():
            self.blur_radius = blur_radius.get()
            option_window.destroy()
            self.update_output_image_preview()  # Update preview with new blur

        tk.Button(option_window, text="OK", command=on_ok).pack(pady=10)
        option_window.bind("<Return>", lambda event: on_ok())
        option_window.wait_window()

    def set_keybindings(self):

        for canvas in [self.canvas, self.canvas2]:
            canvas.bind("<ButtonPress-1>", self.start_box)
            canvas.bind("<B1-Motion>", self.draw_box)
            canvas.bind("<ButtonRelease-1>", self.end_box)
            
            canvas.bind("<Button-3>", self.generate_sam_mask) # Negative point
            canvas.bind("<Button-4>", self.zoom)  # Linux scroll up
            canvas.bind("<Button-5>", self.zoom)  # Linux scroll down
            canvas.bind("<MouseWheel>", self.zoom) #windows scroll
            canvas.bind("<ButtonPress-2>", self.start_pan_mouse)
            canvas.bind("<B2-Motion>", self.pan_mouse)
            canvas.bind("<ButtonRelease-2>", self.end_pan_mouse)
        
        self.root.bind("<c>", lambda event: self.clear_coord_overlay())
        self.root.bind("<d>", lambda event: self.copy_entire_image())
        self.root.bind("<r>", lambda event: self.reset_all())
        self.root.bind("<a>", lambda event: self.add_to_working_image())
        self.root.bind("<s>", lambda event: self.subtract_from_working_image())
        self.root.bind("<w>", lambda event: self.clear_working_image())
        self.root.bind("<p>", self.paint_mode_toggle)
        self.root.bind("<v>", lambda event: self.clear_visible_area())
        self.root.bind("<e>", lambda event: self.edit_image())
        self.root.bind("<u>", lambda event: self.run_whole_image_model("u2net", target_size=320))
        self.root.bind("<i>", lambda event: self.run_whole_image_model("isnet-general-use"))
        self.root.bind("<o>", lambda event: self.run_whole_image_model("rmbg1_4"))
        self.root.bind("<b>", lambda event: self.run_whole_image_model("BiRefNet-general-bb_swin_v1_tiny-epoch_232"))
        self.root.bind("<n>", lambda event: self.run_whole_image_model("BiRefNet-DIS-bb_pvt_v2_b0-epoch_590"))
        self.root.bind("<m>", lambda event: self.run_whole_image_model("BiRefNet-general-bb_swin_v1_tiny-epoch_232_FP16"))        
        if platform.system() == "Darwin":  
            self.root.bind("<Command-z>", lambda event: self.undo())
            self.root.bind("<Command-s>", lambda event: self.save_as_image())
            self.root.bind("<Command-Shift-S>", lambda event: self.quick_save_jpeg())
        else: 
            self.root.bind("<Control-z>", lambda event: self.undo())
            self.root.bind("<Control-s>", lambda event: self.save_as_image())
            self.root.bind("<Control-Shift-S>", lambda event: self.quick_save_jpeg())
        self.root.bind("<Left>", self.pan_left_keyboard)
        self.root.bind("<Right>", self.pan_right_keyboard)
        self.root.bind("<Up>", self.pan_up_keyboard)
        self.root.bind("<Down>", self.pan_down_keyboard)
        
    
    def pan_left_keyboard(self, event):
        self.view_x = max(0, self.view_x - self.pan_step)
        self.update_input_image_preview(Image.NEAREST)
        self.schedule_preview_update()

    def pan_right_keyboard(self, event):
        max_view_x = max(0, self.original_image.width - self.canvas_w / self.zoom_factor)
        self.view_x = min(max_view_x, self.view_x + self.pan_step)
        self.update_input_image_preview(Image.NEAREST)
        self.schedule_preview_update()

    def pan_up_keyboard(self, event):
        self.view_y = max(0, self.view_y - self.pan_step)
        self.update_input_image_preview(Image.NEAREST)
        self.schedule_preview_update()

    def pan_down_keyboard(self, event):
        max_view_y = max(0, self.original_image.height - self.canvas_h / self.zoom_factor)
        self.view_y = min(max_view_y, self.view_y + self.pan_step)
        self.update_input_image_preview(Image.NEAREST)
        self.schedule_preview_update()
    
    
    

    def start_pan_mouse(self, event):
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def pan_mouse(self, event):
        
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.view_x -= dx / self.zoom_factor
            self.view_y -= dy / self.zoom_factor
            
            # Ensure the view stays within the image bounds
            self.view_x = max(0, min(self.view_x, self.original_image.width - self.canvas_w / self.zoom_factor))
            self.view_y = max(0, min(self.view_y, self.original_image.height - self.canvas_h / self.zoom_factor))
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            if hasattr(self, "encoder_output"):
                delattr(self, "encoder_output")
            self.clear_coord_overlay()
            self.model_output_mask = None

            self.update_input_image_preview(Image.NEAREST)
            
    
    def end_pan_mouse(self, event):
        self.panning = False    
        self.update_input_image_preview(Image.BOX)

    def zoom(self, event):
        
        self.pan_step = 30 / self.zoom_factor

        # Calculate mouse position in original image coordinates
        mouse_x = self.view_x + event.x / self.zoom_factor
        mouse_y = self.view_y + event.y / self.zoom_factor
        
        # Update zoom factor
        if event.num == 4 or event.delta>0:  # Zoom in
            self.zoom_factor *= DEFAULT_ZOOM_FACTOR
            self.min_zoom= False
        elif event.num == 5 or event.delta<0:  # Zoom out
            
            self.zoom_factor = max(self.zoom_factor / DEFAULT_ZOOM_FACTOR,
                                   self.lowest_zoom_factor)
            
        # Calculate new view coordinates to keep the zoom centered around the mouse position
        self.view_x = mouse_x - event.x / self.zoom_factor
        self.view_y = mouse_y - event.y / self.zoom_factor
        
        # Ensure the view stays within the image bounds
        self.view_x = max(0, min(self.view_x, self.original_image.width - self.canvas_w / self.zoom_factor))
        self.view_y = max(0, min(self.view_y, self.original_image.height - self.canvas_h / self.zoom_factor))
        
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")


        if self.min_zoom==False: 
            self.update_input_image_preview(resampling_filter=Image.NEAREST)
            if hasattr(self, "encoder_output"):
                delattr(self, "encoder_output")
            self.clear_coord_overlay()

        if self.lowest_zoom_factor == self.zoom_factor: self.min_zoom=True

        self.schedule_preview_update()
        

    def schedule_preview_update(self):
        # Cancel any existing timer and schedule a nicer preview image
        if self.after_id:
            self.root.after_cancel(self.after_id)

        self.after_id = self.root.after(int(self.zoom_delay * 1000), self.update_preview_delayed)

    def update_preview_delayed(self):
        self.update_input_image_preview(resampling_filter=Image.BOX)
        self.after_id = None
    
    def _calculate_preview_image(self, image, resampling_filter):
        # Calculate the size of the visible area in the original image coordinates
        view_width = self.canvas_w / self.zoom_factor
        view_height = self.canvas_h / self.zoom_factor

        left = int(self.view_x)
        top = int(self.view_y)
        right = int(self.view_x + min(math.ceil(view_width), image.width))
        bottom = int(self.view_y + min(math.ceil(view_height), image.height))

        image_to_display = image.crop((left, top, right, bottom))

        image_preview_w = int(image_to_display.width * self.zoom_factor)
        image_preview_h = int(image_to_display.height * self.zoom_factor)

        self.pad_x = max(0, (self.canvas_w - image_preview_w) // 2)
        self.pad_y = max(0, (self.canvas_h - image_preview_h) // 2)

        displayed_image = image_to_display.resize((image_preview_w, image_preview_h), resampling_filter)
        return displayed_image, image_to_display

    #@profile
    def update_input_image_preview(self, resampling_filter=Image.BOX):

        displayed_image, self.orig_image_crop = self._calculate_preview_image(self.original_image, resampling_filter)
        
        if displayed_image.mode == "RGBA":
            image_preview_w, image_preview_h = displayed_image.size
            
            checkerboard = self.checkerboard.crop((0,0,image_preview_w, image_preview_h))
            
            displayed_image = Image.alpha_composite(checkerboard, displayed_image)

        self.input_displayed = displayed_image
        self.tk_image = ImageTk.PhotoImage(self.input_displayed)
        
        self.canvas.delete("all")
        
        self.canvas.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.tk_image)
        self.update_output_image_preview(resampling_filter=resampling_filter)
        
    #@profile
    def update_output_image_preview(self, resampling_filter = Image.BOX):

        if self.show_mask_var.get()==False:
            displayed_image = self.working_image
        else:
            displayed_image = self.working_mask.convert("RGBA")

        # apply background color to full sized image because of the blur background option.
        # otherwise it would be more performant to generate the preview first then apply a solid backgorund colour
        if not self.bg_color.get() == "Transparent":
            displayed_image = self.apply_background_color(displayed_image, 
                                                    self.bg_color.get())
            displayed_image, _ = self._calculate_preview_image(displayed_image, resampling_filter)

        else:
            # calculate the smaller preview first before adding the checkerboard
            displayed_image, _ = self._calculate_preview_image(displayed_image, resampling_filter)

            
            image_preview_w, image_preview_h = displayed_image.size
            
            checkerboard = self.checkerboard.crop((0,0,image_preview_w, image_preview_h))
            
            displayed_image = Image.alpha_composite(checkerboard, displayed_image)
       

        self.output_displayed = displayed_image
        self.outputpreviewtk = ImageTk.PhotoImage(self.output_displayed)
        self.canvas2.delete("all")
        self.canvas2.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.outputpreviewtk)
        
        # zooming out can be laggy especially with multiple scroll wheel clicks, so let tkinter show each zoom out step.
        self.root.update_idletasks()

    def add_undo_step(self):
        self.undo_history_mask.append(self.working_mask.copy())
        if len(self.undo_history_mask) > UNDO_STEPS:
            self.undo_history_mask.pop(0)  
    
    def clear_working_image(self):


        self.canvas2.delete(self.outputpreviewtk)
        self.working_image = Image.new(mode="RGBA",size=self.original_image.size) 
        self.working_mask = Image.new(mode="L",size=self.original_image.size,color=0) 
        self.add_undo_step()
        if hasattr(self, "cached_blurred_shadow"): delattr(self,"cached_blurred_shadow")
        self.update_output_image_preview()
    
    def reset_all(self):
        
        
        self.coordinates=[]
        self.labels=[]

        self.clear_coord_overlay()

        if hasattr(self, "cached_blurred_shadow"):
            delattr(self,"cached_blurred_shadow")

        self.canvas.delete(self.dots)
        if hasattr(self, 'overlay_item'):
            self.canvas.delete(self.overlay_item)
        
        if hasattr(self, "encoder_output"):
            delattr(self, "encoder_output")
            
        self.canvas2.delete(self.outputpreviewtk)
        self.working_image = Image.new(mode="RGBA",size=self.original_image.size) 
        self.working_mask = Image.new(mode="L",size=self.original_image.size,color=0) 
        self.add_undo_step()

        self.update_input_image_preview()
        
    def undo(self):
        
        if len(self.undo_history_mask) > 1:
            self.undo_history_mask.pop() 
            self.working_mask = self.undo_history_mask[-1].copy()  
        
        # this also calls cutout_working_image and update_output_preview
        self.add_drop_shadow()
    
    def copy_entire_image(self):

        self.working_mask = Image.new(mode="L",size=self.original_image.size, color=255) 
        
        self.add_undo_step()
        # this also calls cutout_working_image and update_output_preview
        self.add_drop_shadow()
    
    def cutout_working_image(self):
        empty = Image.new("RGBA", self.original_image.size, 0)
        self.working_image = Image.composite(self.original_image, empty, self.working_mask)

    #@profile
    def _apply_mask_modification(self, operation):

        
        if self.paint_mode.get():
            mask = self.generate_paint_mode_mask()

        else:
            mask = self.model_output_mask

        if mask == None: return

        if self.ppm_var.get():
            #Apply morphological operations to smooth edges
            mask = mask.point(lambda p: p > 128 and 255)  # Binarize the mask
            mask_array = np.array(mask)
            #morphological opening. removes isolated noise and smoothes the boundaries
            mask_array = binary_dilation(mask_array, iterations=1)
            mask_array = binary_erosion(mask_array, iterations=1)
            mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
            
        
        if self.soften_mask_var.get():
            mask = mask.filter(ImageFilter.GaussianBlur(radius=SOFTEN_RADIUS))

        paste_box = (
            int(self.view_x),
            int(self.view_y),
            int(self.view_x + self.orig_image_crop.width),
            int(self.view_y + self.orig_image_crop.height)
        )

        # Create a temporary mask that's the same size as working_mask
        temp_fullsize_mask = Image.new("L", self.working_mask.size, 0)
        temp_fullsize_mask.paste(mask, paste_box)

        self.working_mask = operation(self.working_mask, temp_fullsize_mask)

        self.add_undo_step()
        
        # this function also cuts out the new working image and updates preview
        # even if no shadow is being added
        self.add_drop_shadow()

    def add_to_working_image(self):
        self._apply_mask_modification(ImageChops.add)

    def subtract_from_working_image(self):
        self._apply_mask_modification(ImageChops.subtract)        

    
    def clear_visible_area(self):

        mask_old = self.model_output_mask.copy()
        self.model_output_mask = Image.new("L", self.orig_image_crop.size, 255)
        self.subtract_from_working_image()
        self.model_output_mask = mask_old
    
    def draw_dot(self, x, y,col):

        fill = "red" if col == 1 else "blue"
        radius = 3
        self.dots.append(self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=fill, outline='red'))
        
        # makes sure they are drawn above the overlay
        for i in self.dots:
            self.canvas.tag_raise(i)
    
    def clear_coord_overlay(self):
        self.coordinates=[]
        self.labels=[]
        
        for i in self.dots: self.canvas.delete(i)
        for i in self.lines_id: 
            self.canvas.delete(i)
            
            self.lines=[]
            
        for i in self.lines_id2: 
            
            self.canvas2.delete(i)
            self.lines=[]
        
        if hasattr(self, 'overlay_item'):
            self.canvas.delete(self.overlay_item)
        
        self.model_output_mask = Image.new("L",self.orig_image_crop.size,0)
        #self.mask=None

    def start_box(self, event):
        self.box_start_x = event.x
        self.box_start_y = event.y
        self.box_rectangle = None
        self.box_rectangle2 = None
    
    def draw_box(self, event):
        
        dx = abs(event.x - self.box_start_x)
        dy = abs(event.y - self.box_start_y)

        if self.box_rectangle:
            self.canvas.delete(self.box_rectangle)
        if self.box_rectangle2:
            self.canvas2.delete(self.box_rectangle2)
          
        if dx < MIN_RECT_SIZE and dy < MIN_RECT_SIZE:
            # Even though deleted from canvas it still needs setting to None
            self.box_rectangle = None
            return  
        
        self.box_rectangle = self.canvas.create_rectangle(
            self.box_start_x, self.box_start_y, event.x, event.y, outline="red")
        self.box_rectangle2 = self.canvas2.create_rectangle(
            self.box_start_x, self.box_start_y, event.x, event.y, outline="red")
    
    def end_box(self, event):

        if self.box_rectangle:
            rect_coords = self.canvas.coords(self.box_rectangle)
            self.canvas.delete(self.box_rectangle)
            self.canvas2.delete(self.box_rectangle2)
            
            scaled_coords = [
                (rect_coords[0] - self.pad_x) / self.zoom_factor,
                (rect_coords[1] - self.pad_y) / self.zoom_factor,
                (rect_coords[2] - self.pad_x)  / self.zoom_factor,
                (rect_coords[3] - self.pad_y)  / self.zoom_factor,
            ]
            
            self.box_event(scaled_coords)
        else:
            self.generate_sam_mask(event)
    
    def box_event(self, scaled_coords):
        self._initialise_sam_model()
        
        self.clear_coord_overlay()

        self.coordinates = [[scaled_coords[0], scaled_coords[1]], [scaled_coords[2], scaled_coords[3]]]
        self.labels = [2, 3]  # Assuming 2 for top-left and 3 for bottom-right
        
        self.model_output_mask = self.sam_calculate_mask(self.orig_image_crop, self.sam_encoder, self.sam_decoder, self.coordinates, self.labels)
        
        self.generate_coloured_overlay()

        self.coordinates = []
        self.labels =[]
    
    def canvas_text_overlay(self, overlay_text):
        rect_id = self.canvas.create_rectangle(
            0,
            self.canvas_h // 2 - 30,
            self.canvas_w,
            self.canvas_h // 2 + 30,
            fill="#000000",  # Black
            outline="",
            stipple="gray50" # Add transparency
        )
        canvas_text_id = self.canvas.create_text(
                self.canvas_w // 2,
                self.canvas_h // 2,
                text=overlay_text,
                fill="white",
                font=("Arial", 18, "bold"),
                anchor="center"
            )
        self.canvas.update()
        return canvas_text_id, rect_id

    def load_whole_image_model(self, model_name):
        if not hasattr(self, f"{model_name}_session"):
            self.status_label.config(text=f"Loading {model_name}", fg=STATUS_PROCESSING)
            self.status_label.update()

            self.whole_image_button.configure(text=f"Loading {model_name}"[0:30]+"...", style="Processing.TButton")
            self.whole_image_button.update()

            canvas_text_id, rect_id = self.canvas_text_overlay(f"Loading {model_name}")


            setattr(self, f"{model_name}_session", ort.InferenceSession(f'{MODEL_ROOT}{model_name}.onnx'))

            self.whole_image_button.configure(style="TButton")
            self.whole_image_button.update()

            self.canvas.delete(canvas_text_id)
            self.canvas.delete(rect_id)

        return getattr(self, f"{model_name}_session")
    
    
    
    def run_whole_image_model(self, model_name, target_size=1024):
        
        if self.paint_mode.get():
            return
        
        if model_name == None:
            model_name = self.whole_image_combo.get()
            if model_name=="No Models Found":
                messagebox.showerror("No whole image models found in Models folder","Please see the readme on Github for model download links.")
                return
            target_size = 320 if model_name == "u2net" else 1024

        try:
            session = self.load_whole_image_model(model_name)
        except Exception as e:
            print(e)
            self.status_label.config(text=f"ERROR: {e}", fg=STATUS_PROCESSING)
            self.root.update()
            messagebox.showerror("Error", e)
            self.clear_coord_overlay()
            return

        self.status_label.config(text=f"Processing {model_name}", fg=STATUS_PROCESSING)
        self.status_label.update()

        canvas_text_id, rect_id = self.canvas_text_overlay(f"Processing {model_name}")

        # Trim the text to 30 characters to fit the text box
        self.whole_image_button.configure(text=f"Processing {model_name}"[0:30]+"...", style="Processing.TButton")
        self.whole_image_button.update()

        self.model_output_mask = self.generate_whole_image_model_mask(self.orig_image_crop, session, target_size)

        self.whole_image_button.configure(text=f"Run whole-image model", style="TButton")
        self.whole_image_button.update()

        self.canvas.delete(rect_id)
        self.canvas.delete(canvas_text_id)

        self.generate_coloured_overlay()
     
        
    def generate_whole_image_model_mask(self, image,  session, target_size=1024):
        # adjusted from REMBG.
        input_image = image.convert("RGB").resize((target_size,target_size), Image.BICUBIC)
        

        for i in ["isnet", "rmbg1_4"]:
            if i in os.path.basename(session._model_path):
                std = (1.0, 1.0, 1.0)
                break
            else:
                # u2net, birefnet, rembg2
                std = (0.229, 0.224, 0.225)
        
        if "isnet" in os.path.basename(session._model_path):
            mean = (0.5,0.5,0.5)
        else:    
            mean = (0.485, 0.456, 0.406)

        im_ary = np.array(input_image)
        im_ary = im_ary / np.max(im_ary)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))
        
        input_image =  np.expand_dims(tmpImg, 0).astype(np.float32)
        
        #Inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        start = timer()
        result = session.run([output_name], {input_name: input_image})
        mask = result[0]
        
        
        self.status_label.config(text=f"{round(timer()-start,2)} seconds inference", fg=STATUS_NORMAL)
        
        if "BiRefNet" in os.path.basename(session._model_path):
            def sigmoid(mat):
                # For BiRefNet
                return 1/(1+np.exp(-mat))   
        
            pred = sigmoid(result[0][:, 0, :, :])
    
            ma = np.max(pred)
            mi = np.min(pred)
    
            pred = (pred - mi) / (ma - mi)
            pred = np.squeeze(pred)
    
            mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        
        
        else:
            
            # # Postprocess the mask
            mask = mask.squeeze()  # Remove batch dimension
            mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
        
        mask = mask.convert("L")
        
        return mask


    def generate_coloured_overlay(self):
            
        self.overlay = ImageOps.colorize(self.orig_image_crop.convert("L"), black="blue", white="white") 
        self.overlay.putalpha(self.model_output_mask) 
    
        image_preview_w = int(self.orig_image_crop.width * self.zoom_factor)
        image_preview_h = int(self.orig_image_crop.height * self.zoom_factor)
        
        self.scaled_overlay = self.overlay.resize((image_preview_w, image_preview_h), Image.NEAREST)
        
        self.tk_overlay = ImageTk.PhotoImage(self.scaled_overlay)
        self.overlay_item = self.canvas.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.tk_overlay)

        
        
    # PAINT MODE FUNCTIONS   
    def generate_paint_mode_mask(self):
        img = Image.new('L', (self.orig_image_crop.width, self.orig_image_crop.height), color='black')
        draw = ImageDraw.Draw(img)
        
        
        ellipse_radius = PAINT_BRUSH_DIAMETER/2 / self.zoom_factor
        line_width = int(PAINT_BRUSH_DIAMETER / self.zoom_factor)
        
        for x1, y1, x2, y2 in self.lines:
            
            draw.line((x1, y1, x2, y2), fill='white', width=line_width)
            draw.ellipse((x1-ellipse_radius, y1-ellipse_radius, x1+ellipse_radius, y1+ellipse_radius), fill='white')
            draw.ellipse((x2-ellipse_radius, y2-ellipse_radius, x2+ellipse_radius, y2+ellipse_radius), fill='white')
        
        return img
    
    def paint_mode_toggle(self, event=None):
        
        if event:
            self.ManPaint.toggle()
        
        self.root.update()
#        
        if self.paint_mode.get():
            self.clear_coord_overlay()
            
            for canvas in [self.canvas, self.canvas2]:
                canvas.config(cursor="circle")

                canvas.bind("<ButtonPress-1>", self.paint_draw_point)
                canvas.bind("<B1-Motion>", self.paint_draw_line)
                canvas.bind("<ButtonRelease-1>", self.paint_reset_coords)
        else:    
            self.clear_coord_overlay()

            for canvas in [self.canvas, self.canvas2]:
                canvas.config(cursor="")
                
            self.set_keybindings()

#
    def paint_draw_point(self, event):
        x, y = event.x / self.zoom_factor, event.y / self.zoom_factor
        brush_radius = PAINT_BRUSH_DIAMETER/2
        self.lines_id.append(self.canvas.create_oval(event.x-brush_radius, event.y-brush_radius, 
                                                     event.x+brush_radius, event.y+brush_radius, fill='red'))
        self.lines_id2.append(self.canvas2.create_oval(event.x-brush_radius, event.y-brush_radius, 
                                                       event.x+brush_radius, event.y+brush_radius, fill='red'))
        self.lines.append(((event.x - self.pad_x) / self.zoom_factor,
                            (event.y - self.pad_y) / self.zoom_factor,
                            (event.x - self.pad_x) / self.zoom_factor,
                            (event.y - self.pad_y) / self.zoom_factor,))
    
    def paint_draw_line(self, event):
        
        if self.last_x and self.last_y:
            
            self.lines_id.append(self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                    width=PAINT_BRUSH_DIAMETER, capstyle=tk.ROUND, fill="red"))
            self.lines_id2.append(self.canvas2.create_line(self.last_x, self.last_y, event.x, event.y, 
                                    width=PAINT_BRUSH_DIAMETER, capstyle=tk.ROUND, fill="red"))
            self.lines.append(((self.last_x - self.pad_x) / self.zoom_factor, 
                               (self.last_y - self.pad_y)/ self.zoom_factor, 
                               (event.x - self.pad_x) / self.zoom_factor, 
                               (event.y - self.pad_y) / self.zoom_factor))
        self.last_x, self.last_y = event.x, event.y
        
    def paint_reset_coords(self, event):
        self.last_x, self.last_y = 0, 0
     

    def _initialise_sam_model(self):
        if not hasattr(self, "sam_encoder") or self.sam_model != MODEL_ROOT + self.sam_model_choice.get():
            if self.sam_model_choice.get()=="No Models Found":
                messagebox.showerror("No Segment Anything models found in Models folder","Please see the readme on Github for model download links.")
                self.clear_coord_overlay()
                return
            self.status_label.config(text="Loading model", fg=STATUS_PROCESSING)
            self.status_label.update()
            canvas_text_id, rect_id = self.canvas_text_overlay(f"Loading {self.sam_model_choice.get()}")
            self.sam_model = MODEL_ROOT + self.sam_model_choice.get()
            self.sam_encoder = ort.InferenceSession(self.sam_model + ".encoder.onnx")
            self.sam_decoder = ort.InferenceSession(self.sam_model + ".decoder.onnx")
            self.canvas.delete(canvas_text_id)
            self.canvas.delete(rect_id)
            self.clear_coord_overlay()
            if hasattr(self, "encoder_output"): delattr(self, "encoder_output")
        

    def generate_sam_mask(self, event):

        self._initialise_sam_model()
    
        x, y = (event.x-self.pad_x) / self.zoom_factor, (event.y-self.pad_y) / self.zoom_factor
        self.coordinates.append([x, y])
        
        self.labels.append(event.num if event.num <= 1 else 0)
        
        # Draw dot so user can see responsiveness,
        # as model might take a while to run.
        self.draw_dot(event.x, event.y, event.num)
        self.canvas.update()
        
        self.model_output_mask = self.sam_calculate_mask(self.orig_image_crop, 
                                            self.sam_encoder, self.sam_decoder, self.coordinates, self.labels)
        self.generate_coloured_overlay()
        
        # Repeated to ensure the dot stays on top
        self.draw_dot(event.x, event.y, event.num)

    def sam_calculate_mask(self, img, sam_encoder, sam_decoder, coordinates, labels,):

        target_size = 1024
        input_size = (684, 1024)
        encoder_input_name = sam_encoder.get_inputs()[0].name
     
        img = img.convert("RGB")
        cv_image = np.array(img)
        original_size = cv_image.shape[:2]

        scale_x = input_size[1] / cv_image.shape[1]
        scale_y = input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)

        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )

        cv_image = cv2.warpAffine(
            cv_image,
            transform_matrix[:2],
            (input_size[1], input_size[0]),
            flags=cv2.INTER_LINEAR,
        )

        ## encoder
        status=""

        if not hasattr(self, "encoder_output"):
            encoder_inputs = {
                encoder_input_name: cv_image.astype(np.float32),
            }
                     
            self.status_label.config(text="Calculating Image Embedding. May take a while", fg=STATUS_PROCESSING)
            canvas_text_id, rect_id = self.canvas_text_overlay(f"Calculating Image Embedding")

            self.root.update()
            
            start = timer()
            self.encoder_output = sam_encoder.run(None, encoder_inputs)

            status = f"{round(timer()-start,2)} seconds to calculate embedding | "

            self.status_label.config(text=status, fg=STATUS_NORMAL)
            self.canvas.delete(canvas_text_id)
            self.canvas.delete(rect_id)
            
        self.root.update()
        
        image_embedding = self.encoder_output[0]
        
        input_points = np.array(coordinates)
        input_labels = np.array(labels)
        
        onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[
            None, :, :
        ]
        onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[
            None, :
        ].astype(np.float32)
        onnx_coord = self.apply_coords(onnx_coord, input_size, target_size).astype(
            np.float32
        )

        onnx_coord = np.concatenate(
            [
                onnx_coord,
                np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32),
            ],
            axis=2,
        )
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(input_size, dtype=np.float32),
        }
        start=timer()
        masks, a, b = sam_decoder.run(None, decoder_inputs)
        
        status += f"{round(timer()-start,2)} seconds inference | "
        status += f"{round(float(a[0][0]),2)} confidence score"

        self.status_label.config(text=status)

        inv_transform_matrix = np.linalg.inv(transform_matrix)
        masks = self.transform_masks(masks, original_size, inv_transform_matrix)

        mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
        for m in masks[0, :, :, :]:
            mask[m > 0.0] = [255, 255, 255]

        mask = Image.fromarray(mask).convert("L")

        return mask
    
    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int):
        """
            Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)

        return (newh, neww)


    def apply_coords(self,coords: np.ndarray, original_size, target_length):
        """
           Expects a numpy array of length 2 in the final dimension. Requires the
           original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], target_length
        )

        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        return coords

    def transform_masks(self, masks, original_size, transform_matrix):
        output_masks = []

        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)

        return np.array(output_masks)
   
      
    def quick_save_jpeg(self):

        self.status_label.config(text="", fg=STATUS_NORMAL)

        file_path = self.image_paths[self.current_image_index]

        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)

        file_name_nobg = os.path.splitext(file_name)[0] + "_nobg.jpg"

        user_filename = asksaveasfilename(title = "Save as", 
                                          defaultextension=".jpg", filetypes=[("jpg", ".jpg")],
                                          
                                          initialdir=dir_path, initialfile=file_name_nobg,
                                          )

        if len(user_filename) == 0:
            return None
        
        if not user_filename.endswith(".jpg"): user_filename+=".jpg"
        
        self.add_drop_shadow()
        workimg = self.working_image
        workimg = self.apply_background_color(workimg, "White")
        workimg = workimg.convert("RGB")
        if self.image_exif:
            workimg.save(user_filename, quality=95, exif=self.image_exif)
        else:
            workimg.save(user_filename, quality=95)
        print("Saved to "+ user_filename)
        self.status_label.config(text="Saved to "+ user_filename)
        self.canvas.update()

    
    def show_save_options(self):
        option_window = tk.Toplevel(self.root)
        option_window.title("Save Options")
        option_window.geometry("300x280")  
        option_window.resizable(False, False)
        option_window.transient(self.root)
        
        file_type = tk.StringVar(value=self.save_file_type)
        quality = tk.IntVar(value=self.save_file_quality)
        
        def update_quality_state(event=None):
            if file_type.get() in ["lossy_webp", "jpg"]:
                quality_slider.config(state="normal")
                quality_label.config(state="normal")
                quality_value_label.config(state="normal")
            else:
                quality_slider.config(state="disabled")
                quality_label.config(state="disabled")
                quality_value_label.config(state="disabled")
        
        def update_quality_value(event=None):
            quality_value_label.config(text=str(quality.get()))
        
        tk.Label(option_window, text="Select file type:").pack(anchor="w", padx=10, pady=(10, 5))
        
        radio_buttons = []
        for text, value in [("PNG", "png"), ("Lossless WebP", "lossless_webp"), 
                            ("Lossy WebP", "lossy_webp"), ("JPEG", "jpg")]:
            rb = tk.Radiobutton(option_window, text=text, variable=file_type, value=value)
            rb.pack(anchor="w", padx=20)
            rb.bind("<ButtonRelease-1>", update_quality_state)
            radio_buttons.append(rb)
        
        quality_label = tk.Label(option_window, text="Quality (for Lossy WebP and JPEG):")
        quality_label.pack(anchor="w", padx=10, pady=(10, 0))
        
        quality_frame = ttk.Frame(option_window)
        quality_frame.pack(fill="x", padx=10)
        
        quality_slider = ttk.Scale(quality_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=quality, command=update_quality_value)
        quality_slider.pack(side="left", fill="x", expand=True)
        
        quality_value_label = ttk.Label(quality_frame, text=str(quality.get()), width=3)
        quality_value_label.pack(side="left", padx=(5, 0))

        save_mask_var = tk.BooleanVar(value=False)
        save_mask_checkbox = tk.Checkbutton(option_window, text="Save Mask", variable=save_mask_var)
        save_mask_checkbox.pack(anchor="w", padx=10, pady=(10, 0))

         # Checkbox explanation label
        mask_explanation_label = tk.Label(option_window, text="(appends _mask.png to filename)", font=("TkDefaultFont", 8))
        mask_explanation_label.pack(anchor="w", padx=10)
        
        update_quality_state() 
        
        result = {"file_type": None, "quality": None}
        
        def on_ok():
            result["file_type"] = file_type.get()
            result["quality"] = quality.get()
            self.save_file_type = file_type.get()
            self.save_file_quality = quality.get()
            self.save_mask = save_mask_var.get()
            option_window.destroy()
        
        tk.Button(option_window, text="OK", command=on_ok).pack(pady=10)
        option_window.bind("<Return>", lambda event: on_ok())
        option_window.wait_window()
        
        return result
    
    def save_as_image(self):
        options = self.show_save_options()
        if not options["file_type"]:
            return

        file_types = {
            "png": (".png", "PNG files"),
            "lossless_webp": (".webp", "WebP files"),
            "lossy_webp": (".webp", "WebP files"),
            "jpg": (".jpg", "JPEG files")
        }

        ext, file_type = file_types[options["file_type"]]
        
        initial_file = os.path.splitext(os.path.basename(self.image_paths[self.current_image_index]))[0] +"_nobg" + ext

        user_filename = asksaveasfilename(
            title="Save as",
            defaultextension=ext,
            filetypes=[(file_type, "*" + ext)],
            initialdir=os.path.dirname(self.image_paths[self.current_image_index]),
            initialfile=initial_file
        )

        if not user_filename:
            return

        canvas_text_id, rect_id = self.canvas_text_overlay("Saving Image")

        if not user_filename.lower().endswith(ext):
            user_filename += ext
        
        # this also calls cutout_working_image and update_output_preview
        self.add_drop_shadow()
        workimg = self.working_image

        if not self.bg_color.get() == "Transparent":
            workimg = self.apply_background_color(workimg, self.bg_color.get())


        save_params = {}
        if self.image_exif:
            save_params['exif'] = self.image_exif

        if options["file_type"] == "png":
            save_params['optimize'] = True
        elif options["file_type"] == "lossless_webp":
            save_params['lossless'] = True
        elif options["file_type"] == "lossy_webp":
            save_params['lossless'] = False
            save_params['quality'] = options["quality"]
        elif options["file_type"] == "jpg":
            save_params['quality'] = options["quality"]
            workimg = workimg.convert("RGB")

        self.status_label.config(text=f"Saving to {ext.upper()[1:]}", fg=STATUS_PROCESSING)
        self.root.update()

        try:
            workimg.save(user_filename, **save_params)
            print(f"Saved to {user_filename}")
            self.status_label.config(text=f"Saved to {user_filename}", fg=STATUS_NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
        if self.save_mask:
            base_name, _ = os.path.splitext(user_filename)
            mask_filename = base_name + "_mask.png"
            self.working_mask.save(mask_filename, "PNG")
            print(f"Mask saved to {mask_filename}")
        self.canvas.delete(canvas_text_id)
        self.canvas.delete(rect_id)
        self.canvas.update()

    

    def update_bg_color(self):
                
        if self.bg_color_var.get() == 1:
            self.bgcolor = (255, 255, 255, 255)

        else:
            self.bgcolor = None
            
        self.update_output_image_preview()

    #@profile
    def apply_background_color(self, img, color):

        if color == "Blurred_(Slow)":
            s=timer()
            
            cv_original = np.array(self.original_image)
            cv_mask = np.array(self.working_mask)

            # bg remover masks tend to be slightly smaller than object
            # so we need to expand the mask first to catch the pixels outside of
            # the cutout object, then inpaint using background pixels
            kernel = np.ones((5,5), np.uint8)  
            expanded_mask = cv2.dilate(cv_mask, kernel, iterations=1)  

            # (3,3) kernel with 3 iterations: expands 3 pixels out, but more gradually
            # (7,7) kernel with 1 iteration: expands 3 pixels out all at once, might be more blocky
            # Both reach 3 pixels out, but the first approach gives smoother edges

            filled_background = cv2.inpaint(cv_original, expanded_mask, 3, cv2.INPAINT_TELEA)

            blur_radius = int(self.blur_radius)
            if blur_radius % 2 == 0:
                blur_radius += 1
            blurred_background = cv2.blur(filled_background, (blur_radius, blur_radius))
            
            colored_image = Image.fromarray(blurred_background)
            colored_image.paste(self.working_image, mask=self.working_mask)
            print(f"{timer()-s} to calculate blurred background")
        else:
            colored_image = Image.new("RGBA", img.size, color)
            colored_image = Image.alpha_composite(colored_image, img)

        return colored_image
    
    def on_closing(self):
        
        print("closing")
        self.root.destroy()
        
    def edit_image(self):
        if messagebox.askyesno("Continue to edit the image?", 
                               """Editing the original image will reset the progress on your output image. 

Consider saving the current mask first (Save As menu). Would you like to edit the image?"""): 

            editor = ImageEditor(self.root, self.original_image, self.file_count)
            self.root.wait_window(editor.crop_window)
            if editor.final_image:
                self.original_image = editor.final_image
                self.initialise_new_image()
        
    def load_clipboard(self):
        
        img = ImageGrab.grabclipboard()

        if "PIL." in str(type(img)):
            self.original_image = img
            
            self.initialise_new_image()

            self.image_paths = [os.getcwd()+"/clipboard_image.png"]
            self.current_image_index=0

            

            self.NextImg.configure(state=tk.DISABLED)
            self.root.title("Background Remover") # Reset title

            self.status_label.config(text=f"File will be saved to {os.getcwd()+"/clipboard_image.png"}")
        else:
            #TODO if clipboard contains a path to an image, load it
            
            messagebox.showerror("Error", f"No image found on clipboard.\nClipboard contains {type(img)}")
        
    def load_image_from_dialog(self):
        image_path = askopenfilename(
            title="Select an Image",
            filetypes=pillow_formats
        )
        if not image_path:
            return
        
        self.original_image = Image.open(image_path)
        self.original_image = ImageOps.exif_transpose(self.original_image)

        self.image_exif = self.original_image.info.get('exif')
        print("EXIF data found" if self.image_exif else "No EXIF data found.")
       
        self.initialise_new_image()


        self.image_paths = [image_path]
        self.current_image_index = 0 # Indicates not from list
        self.NextImg.configure(state=tk.DISABLED)
        self.root.title("Background Remover") # Reset title
        self.status_label.config(text=f"Loaded image: {os.path.basename(image_path)}", fg=STATUS_NORMAL)


    def add_drop_shadow(self):
        
        if not self.enable_shadow_var.get():
            self.cutout_working_image()
            self.update_output_image_preview()  
            return False

        shadow_opacity = self.shadow_opacity_slider.get()
        shadow_radius = int(self.shadow_radius_slider.get())
        shadow_x = int(self.shadow_x_slider.get())
        shadow_y = int(self.shadow_y_slider.get())

        alpha = self.working_mask

        # Downsample for performance
        original_size = alpha.size
        downsample_factor = 0.5  
        new_size = (int(original_size[0] * downsample_factor), int(original_size[1] * downsample_factor))
        alpha_resized = alpha.resize(new_size, Image.NEAREST)

        blurred_alpha_resized = alpha_resized.filter(ImageFilter.GaussianBlur(radius=shadow_radius * downsample_factor))

        blurred_alpha = blurred_alpha_resized.resize(original_size, Image.NEAREST)

        shadow_opacity_alpha = blurred_alpha.point(lambda p: int(p * shadow_opacity *((shadow_radius/10)+1))) # multiply by 2 to darken more when blur radius is high

        shadow_image = Image.new("RGBA", self.working_image.size, (0, 0, 0, 0))
        shadow_image.putalpha(shadow_opacity_alpha)

        shadow_with_offset = Image.new("RGBA", self.working_image.size, (0, 0, 0, 0))
        shadow_with_offset.paste(shadow_image, (shadow_x, shadow_y), shadow_image)

        self.cutout_working_image()

        self.working_image = Image.alpha_composite(shadow_with_offset, self.working_image)

        self.update_output_image_preview()

        return True

    def toggle_shadow_options(self):
        if self.enable_shadow_var.get():
            # Show shadow options when checkbox is checked
            self.shadow_opacity_label.pack(fill="x")
            self.shadow_opacity_slider.pack(fill="x")
            self.shadow_x_label.pack(fill="x")
            self.shadow_x_slider.pack(fill="x")
            self.shadow_y_label.pack(fill="x")
            self.shadow_y_slider.pack(fill="x")
            self.shadow_radius_label.pack(fill="x")
            self.shadow_radius_slider.pack(fill="x")
            self.shadow_options_frame.pack(fill="x", padx=5, pady=5)
        else:
            self.shadow_opacity_label.pack_forget()
            self.shadow_opacity_slider.pack_forget()
            self.shadow_x_label.pack_forget()
            self.shadow_x_slider.pack_forget()
            self.shadow_y_label.pack_forget()
            self.shadow_y_slider.pack_forget()
            self.shadow_radius_label.pack_forget()
            self.shadow_radius_slider.pack_forget()
            self.shadow_options_frame.pack_forget()
        self.add_drop_shadow()
    

    def load_next_image(self):
        """Loads the next image from the command-line argument list."""
        if not self.image_paths or self.current_image_index >= len(self.image_paths) - 1:
            print("No more images in the list.")
            self.NextImg.configure(state=tk.DISABLED) 
            return

        self.current_image_index += 1
        next_image_path = self.image_paths[self.current_image_index]

        print(f"Loading next image: {next_image_path}")

        try:
            new_image = Image.open(next_image_path)
            self.original_image = ImageOps.exif_transpose(new_image)
            
            self.image_exif = self.original_image.info.get('exif')
            print("EXIF data found" if self.image_exif else "No EXIF data found.")

            self.initialise_new_image()

            self.file_count = f' - Image {self.current_image_index + 1} of {len(self.image_paths)}'
            self.root.title("Background Remover" + self.file_count)

            if self.current_image_index >= len(self.image_paths) - 1:
                self.NextImg.configure(state=tk.DISABLED)

            self.status_label.config(text=f"Loaded image {self.current_image_index + 1} of {len(self.image_paths)}", fg=STATUS_NORMAL)

        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Could not load image:\n{next_image_path}\n\nError: {e}")

            self.status_label.config(text=f"Error loading {next_image_path}", fg=STATUS_PROCESSING)




    def initialise_new_image(self):
        
        self.canvas2.delete("all")
        
        self.setup_image_display()
        self.update_input_image_preview()
        self.clear_coord_overlay()
        self.reset_all()

    def load_mask(self):
        initial_file = os.path.splitext(os.path.basename(self.image_paths[self.current_image_index]))[0] + "_mask.png"
        mask_path = askopenfilename(
            title="Select a Mask",
            filetypes=[("PNG files", "*.png")],
            initialdir=os.path.dirname(self.image_paths[self.current_image_index]),
            initialfile=initial_file
        )
        if not mask_path:
            return

        try:
            mask_image = Image.open(mask_path)
            if mask_image.mode != "L":
                messagebox.showerror("Error", "The selected file is not a grayscale image.")
                return

            if mask_image.size != self.original_image.size:
                messagebox.showerror("Error", "The mask dimensions do not match the original image dimensions.")
                return

            self.working_mask = mask_image
            self.add_undo_step()
            self.add_drop_shadow()

            print(f"Mask loaded from {mask_path}")
            self.status_label.config(text=f"Mask loaded from {mask_path}", fg=STATUS_NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask: {str(e)}")

        self.canvas.update()
        
    def show_help(self):
        message = """
            Interactive Background Remover by Sean (Prickly Gorse)

A user interface for removing backgrounds using interactive models (Segment Anything) and Whole Image Models (u2net, disnet, rmbg, BirefNet)

Load your image, and either run one of the whole image models (u2net <u>, disnet <i>, rmbg <o>, BiRefNet) or click/draw a box to run Segment Anything. Left click is a positive point, right is a negative (avoid this area) point.

The original image is displayed on the left, the current image you are working on is displayed to the right.

Type A to add the current mask to your image, S to subtract.

Scroll wheel to zoom, and middle click to pan around the image. The models will be applied only to the visible zoomed image, which enables much higher detail and working in finer detail than just running the models on the whole image

Use paintbrush mode <p> to draw areas that you want to add/remove to the image without using a model.

Post process mask removes the partial transparency from outputs of whole-image models. 

Includes a built in image editor and cropper. Using this will reset your current working image.

Usage:

Left Mouse click: Add coordinate point for segment anything models
Right Mouse click: Add negative coordinate (area for the model to avoid)
Left click and drag: Draw box for segment anything models

Hotkeys:

<a> Add current mask to working image
<s> Subtract current mask from working image
<Ctrl+z> Undo last action
<p> Manual paintbrush mode
<c> Clear current mask (and coordinate points)
<w> Reset the current working image
<r> Reset everything (image, masks, coordinates)
<v> Clear the visible area on the working image
<Ctrl+S> Save as....
<Ctrl+Shift+S> Quick save JPG with white background

Whole image models (if downloaded to Models folder)
<u> u2net
<i> disnet
<o> rmbg1.4
<b> BiRefNet-general-bb_swin_v1_tiny-epoch_232

            """
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Help/About")
        info_window.geometry("800x800")
        
        frame = tk.Frame(info_window)
        frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        text_widget = tk.Text(frame, wrap=tk.WORD)
        text_widget.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        scrollbar = tk.Scrollbar(frame, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget.config(yscrollcommand=scrollbar.set)

        text_widget.insert(tk.END, message)
        text_widget.config(state=tk.DISABLED)
        
        close_button = tk.Button(info_window, text="Close", command=info_window.destroy)
        close_button.pack(pady=10)


class ImageEditor:
    def __init__(self, parent, original_image, file_count):

        self.parent = parent
        self.file_count = file_count
        self.original_image = original_image
        self.rotated_original = original_image
        self.total_rotation = 0
        self.build_crop_gui()

        
    def build_crop_gui(self):
   
        self.crop_window = tk.Toplevel()
        self.crop_window.title("Preprocess and Crop Image"+self.file_count)
        
        # Force it on top
        self.crop_window.lift()
        self.crop_window.focus_force()
        self.crop_window.grab_set()
        self.crop_window.grab_release()
        
        self.original_image = self.original_image
        self.rotated_original = self.original_image
        

        # Get the usable screen dimensions
        m = [m for m in get_monitors() if m.is_primary][0]

        screen_width = (m.width -350)
        screen_height = m.height-100 
        # subtract 100 because maximising window doesnt work until __init__ finished,
        # so we have to start application using full screen dimensions
        
        if platform.system() == "Windows":
            self.crop_window.state('zoomed')
        elif platform.system() == "Darwin":
            self.crop_window.state('zoomed') #unsure if works
        else:
            self.crop_window.attributes('-zoomed', True)

                
        self.image_ratio = min(screen_width / self.original_image.width, screen_height / self.original_image.height)
        self.scaled_width = int(self.original_image.width * self.image_ratio)
        self.scaled_height = int(self.original_image.height * self.image_ratio)
        
        self.display_image = self.original_image.resize((self.scaled_width, self.scaled_height))
        self.crop_window_tk_image = ImageTk.PhotoImage(self.display_image)
        

        self.canvas_crop_window = Canvas(self.crop_window, width=screen_width, height=screen_height)
        self.canvas_crop_window.pack(side=tk.LEFT)
        
        #self.canvas_crop_window.create_image(screen_width/2, screen_height/2, image=self.crop_window_tk_image)
        
        
        self.slider_frame = tk.Frame(self.crop_window, width=300)
        self.slider_frame.pack(side=tk.LEFT)
        
        
        self.croplabel = ttk.Label(self.slider_frame)
        self.croplabel.configure(text='Click and drag to draw a crop box')
        self.croplabel.pack(pady=5)
        
        
        self.sliders = {}
        slider_params = {
            'highlight': (0.1, 2.0, 1.0),
            'midtone': (0.1, 2.0, 1.0),
            'shadow': (0.1, 3.0, 1.0),
            'tone_curve': (0.01, 0.5, 0.1),
            'brightness': (0.1, 2.0, 1.0),
            'contrast': (0.1, 2.0, 1.0),
            'saturation': (0.1, 2.0, 1.0),
            'white_balance': (2000, 10000, 6500),
            'unsharp_radius': (0.1, 50, 1.0),
            'unsharp_amount': (0, 500, 0),
            'unsharp_threshold': (0, 255, 0)
        }
        
        
        
        label_width = 15

        for param, (min_val, max_val, default) in slider_params.items():
            frame = tk.Frame(self.slider_frame)
            frame.pack(fill=tk.X, pady=2)  

            label = ttk.Label(frame, text=param.replace("_"," ").capitalize(), width=label_width, anchor="w")
            label.pack(side=tk.LEFT, padx=(5, 10))  #

            # Slider widget
            self.sliders[param] = tk.Scale(
                frame,
                from_=min_val,
                to=max_val,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                length=300,   
                width=20
            )
            self.sliders[param].set(default)
            self.sliders[param].pack(side=tk.LEFT) 
                
        # Add the command after all sliders are created and set
        # to avoid the function being called
        for slider in self.sliders.values():
            slider.config(command=self.update_crop_preview)
            
        self.rotation_frame = Frame(self.slider_frame)
        self.rotation_frame.pack(pady=10)
           
        self.rotate_left_button = Button(self.rotation_frame, text="Rotate Left", command=lambda: self.rotate_image(90))
        self.rotate_left_button.pack(side=tk.LEFT, padx=5)
           
        self.rotate_right_button = Button(self.rotation_frame, text="Rotate Right", command=lambda: self.rotate_image(-90))
        self.rotate_right_button.pack(side=tk.LEFT, padx=5)

        self.reset_frame = Frame(self.slider_frame)
        self.reset_frame.pack(pady=10)
        self.reset_button = Button(self.reset_frame, text="Reset to Original", command=self.reset_sliders)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        self.common_adjustments = Button(self.reset_frame, text="Use Preset", command=self.common_slider_adjustment)
        self.common_adjustments.pack(side=tk.LEFT, padx=5)
        
        
        self.crop_button = tk.Button(self.slider_frame, text="Continue with this image", command=self.crop_image)
        self.crop_button.pack(fill=tk.X, pady=10)

         
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rectangle_crop_window = None
        
        self.canvas_crop_window.bind("<ButtonPress-1>", self.crop_canvas_on_press)
        self.canvas_crop_window.bind("<B1-Motion>", self.crop_canvas_on_drag)
        self.canvas_crop_window.bind("<ButtonRelease-1>", self.crop_canvas_on_release)
        
        self.crop_window.bind("<Return>", lambda x: self.crop_image())

        self.crop_window.update_idletasks()
        self.crop_window.update()
        self.update_crop_preview()
    
    def reset_sliders(self):
         default_values = {
             'highlight': 1.0,
             'midtone': 1.0,
             'shadow': 1.0,
             'brightness': 1.0,
             'contrast': 1.0,
             'saturation': 1.0,
             'tone_curve': 0.1,
             'white_balance': 6500,
             'unsharp_radius': 1.0,
             'unsharp_amount': 0,
             'unsharp_threshold': 0
         }
         for param, value in default_values.items():
             self.sliders[param].set(value)
         self.rotate_image(-self.total_rotation)
    
    def common_slider_adjustment(self):
         default_values = {
             'highlight': 0.75,
             'midtone': 0.85,
             'shadow': 1.5,
             'brightness': 1.50,
             'contrast': 1.25,
             'saturation': 1.07,
             'tone_curve': 0.1,
             'white_balance': 7000,
             'unsharp_radius': 1.0,
             'unsharp_amount': 1.0,
             'unsharp_threshold': 3
         }
         for param, value in default_values.items():
             self.sliders[param].set(value)     

    
    def update_crop_preview(self, *args):
        params = {param: self.sliders[param].get() for param in self.sliders}
        img_adjusted = self.adjust_image_levels(self.display_image, **params)
        
        self.tk_image = ImageTk.PhotoImage(img_adjusted)
        self.canvas_crop_window.delete("all")  # Clear the canvas
        self.canvas_crop_window.create_image( self.canvas_crop_window.winfo_width()/2,
                                             self.canvas_crop_window.winfo_height()/2,
                                             image=self.tk_image)
        
        self.canvas_crop_window.create_rectangle(
            self.canvas_crop_window.winfo_width()/2 - img_adjusted.width/2,
            self.canvas_crop_window.winfo_height()/2 - img_adjusted.height/2,
            self.canvas_crop_window.winfo_width()/2 + img_adjusted.width/2 -1,
            self.canvas_crop_window.winfo_height()/2 + img_adjusted.height/2 -1,
            outline="black"
        )

        
        if self.rectangle_crop_window:
            self.rectangle_crop_window = self.canvas_crop_window.create_rectangle(
                self.start_x, self.start_y, self.end_x, self.end_y, outline="red")
     



    def crop_canvas_on_press(self, event):
          self.start_x, self.start_y = event.x, event.y
          
          if self.rectangle_crop_window:
              self.canvas_crop_window.delete(self.rectangle_crop_window)
          self.rectangle_crop_window = self.canvas_crop_window.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
          
    def crop_canvas_on_drag(self, event):
          # Update the size of the rectangle as the mouse is dragged
          cur_x, cur_y = event.x, event.y
          self.canvas_crop_window.coords(self.rectangle_crop_window, self.start_x, self.start_y, cur_x, cur_y)
          
    def crop_canvas_on_release(self, event):
          self.end_x, self.end_y = event.x, event.y

    def crop_image(self):
        if self.rectangle_crop_window and self.end_x is not None and self.end_y is not None:
            # Convert the coordinates back to the original image dimensions
            x1,y1,x2,y2 = self.canvas_crop_window.coords(self.rectangle_crop_window)
            
            x1 = x1 - (self.canvas_crop_window.winfo_width()/2) + self.display_image.width/2
            x2 = x2 - (self.canvas_crop_window.winfo_width()/2) + self.display_image.width/2
            y1 = y1 - (self.canvas_crop_window.winfo_height()/2) + self.display_image.height/2
            y2 = y2 - (self.canvas_crop_window.winfo_height()/2) + self.display_image.height/2
            
            orig_start_x = int(x1 / self.image_ratio)
            orig_start_y = int(y1 / self.image_ratio)
            orig_end_x = int(x2 / self.image_ratio)
            orig_end_y = int(y2 / self.image_ratio)
            
            # Crop the rotated image
            self.original_image = self.rotated_original.crop((orig_start_x, orig_start_y, orig_end_x, orig_end_y))
        else:
            # If no crop box, use the entire rotated image
            self.original_image = self.rotated_original
    
        # Apply color adjustments
        params = {param: self.sliders[param].get() for param in self.sliders}
        self.final_image = self.adjust_image_levels(self.original_image, **params)

        self.crop_window.destroy()


    def rotate_image(self, angle):
        self.canvas_crop_window.delete("all")
        
        self.total_rotation = (self.total_rotation + angle) % 360
        
        self.rotated_original = self.original_image.rotate(self.total_rotation, expand=True)
        
        # Resize the rotated image to fit the screen
        screen_width = self.crop_window.winfo_width() - 350
        screen_height = self.crop_window.winfo_height() - 100
        self.image_ratio = min(screen_width / self.rotated_original.width, screen_height / self.rotated_original.height)
        self.scaled_width = int(self.rotated_original.width * self.image_ratio)
        self.scaled_height = int(self.rotated_original.height * self.image_ratio)
        
        self.display_image = self.rotated_original.resize((self.scaled_width, self.scaled_height))
        
        # Remove any existing crop box
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rectangle_crop_window = None
        
        self.update_crop_preview()

    def adjust_image_levels(self, image, highlight, midtone, shadow, brightness, contrast, 
                            saturation, tone_curve, white_balance,
                            unsharp_radius, unsharp_amount, unsharp_threshold):

        img_array = np.array(image)


        if white_balance != 6500:
            img_array = self.adjust_white_balance(img_array, white_balance)

        # Combine all masks into a single operation
        x = np.arange(256, dtype=np.float32)
        highlight_mask = self.smooth_transition(x, 192, tone_curve)
        shadow_mask = 1 - self.smooth_transition(x, 64, tone_curve)
        midtone_mask = 1 - highlight_mask - shadow_mask
    
        # Create a lookup table
        lut = (x * highlight * highlight_mask +
            x * midtone * midtone_mask +
            x * shadow * shadow_mask).clip(0, 255).astype(np.uint8)
    

        adjusted = lut[img_array]

    
        adjusted_image = Image.fromarray(adjusted)
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(adjusted_image)
            adjusted_image = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(adjusted_image)
            adjusted_image = enhancer.enhance(contrast)
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(adjusted_image)
            adjusted_image = enhancer.enhance(saturation)
                
        if unsharp_amount>0:
            adjusted_image=adjusted_image.filter(ImageFilter.UnsharpMask(radius = int(unsharp_radius), percent = int(unsharp_amount), threshold = int(unsharp_threshold))) 
     
        return adjusted_image

    
    def adjust_white_balance(self, img_array, temperature):
        rgb = self.kelvin_to_rgb(temperature)
        r_factor, g_factor, b_factor = [x / max(rgb) for x in rgb]
    
        #img_array = np.array(image)
    
        avg_brightness = np.mean(img_array[:,:,:3])
    
        img_array[:,:,0] = np.clip(img_array[:,:,0] * r_factor, 0, 255)
        img_array[:,:,1] = np.clip(img_array[:,:,1] * g_factor, 0, 255)
        img_array[:,:,2] = np.clip(img_array[:,:,2] * b_factor, 0, 255)
    
        new_avg_brightness = np.mean(img_array[:,:,:3])
    
        brightness_factor = avg_brightness / new_avg_brightness
        img_array[:,:,:3] = np.clip(img_array[:,:,:3] * brightness_factor, 0, 255)
    
        #adjusted = Image.fromarray(img_array.astype('uint8'), mode=image.mode)
    
        return img_array


    def kelvin_to_rgb(self, kelvin):
        temp = kelvin / 100
        
        if temp <= 66:
            red = 255
            green = temp
            green = 99.4708025861 * math.log(green) - 161.1195681661
            if temp <= 19:
                blue = 0
            else:
                blue = temp - 10
                blue = 138.5177312231 * math.log(blue) - 305.0447927307
        else:
            red = temp - 60
            red = 329.698727446 * math.pow(red, -0.1332047592)
            green = temp - 60
            green = 288.1221695283 * math.pow(green, -0.0755148492)
            blue = 255

        return (max(0, min(255, red)),
                max(0, min(255, green)),
                max(0, min(255, blue)))
     
     
    def smooth_transition(self, x, center, steepness):
         return expit((x - center) / (255 * steepness))

if __name__ == "__main__":


    files_to_process = sys.argv[1:] 

    if not files_to_process:
        
        initial_file = askopenfilename(
            title="Select an Image to Start",
            filetypes=pillow_formats
        )
        if initial_file:
            files_to_process = [initial_file]



    root = tk.Tk()
    app = BackgroundRemoverGUI(root, files_to_process)
    root.mainloop()

