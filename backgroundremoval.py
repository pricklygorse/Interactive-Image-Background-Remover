#!/usr/bin/env python3
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import Canvas, Frame, Button, messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageEnhance, ImageGrab
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

DEFAULT_ZOOM_FACTOR = 1.5
MODEL_ROOT = "./Models/"
STATUS_PROCESSING = "#f00"
STATUS_NORMAL = "#000000"
PAINT_BRUSH_DIAMETER = 18



class ImageClickApp:
    def __init__(self, root, image_path, file_count=""):
        
        self.file_count  = file_count
        self.save_file = image_path[0:-4]+"_nobg.png"
        self.save_file_jpg = image_path[0:-4]+"_nobg.jpg"
        self.save_file_type = "png"
        self.save_file_quality = 90
        self.coordinates = []
        self.labels=[]
        self.bgcolor = None
        self.dots=[]
        
        # For segment anything models, to stop the user accidently drawing a tiny rectangle
        # instead of clicking to make a point
        self.min_rect_size = 5
        
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.last_x, self.last_y = 0, 0
        self.pan_step = 30
        

        
        self.lines = []
        self.lines_id = []
        self.lines_id2 = []
        
        self.original_image = Image.open(image_path)
        self.original_image = ImageOps.exif_transpose(self.original_image)
        try:
            self.image_exif = self.original_image.info['exif']
            print("EXIF data found")
        except KeyError:
            self.image_exif = None
            print("No EXIF data found.")

        
        self.root = root
        self.root.title("Background Remover"+file_count)
        
        
        s = ttk.Style()
        s.theme_use('alt')
        




        m = [m for m in get_monitors() if m.is_primary][0]
        
        self.canvas_w = (m.width -300) //2
        self.canvas_h = m.height-100 

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
            self.root.state('zoomed') #unsure if works
        else:
            self.root.attributes('-zoomed', True)

        self.root.update_idletasks()
        self.build_gui()
        self.update_zoomed_view()


        self.mask = Image.new("L", (int(self.orig_image_crop.width), 
                                            int(self.orig_image_crop.height)),0)

        self.set_keybindings()


        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        

    def on_resize(self, event):
        # Check if window has changed dimensions as this function can get called
        # multiple times for some reason, even just when clicking the canvas

        self.root.update_idletasks()
        if not self.root.winfo_height() == self.init_height or not self.root.winfo_width() == self.init_width:
            
            # The currently displayed preview that is being used will change, so 
            # a new encoding has to be calculated
            if hasattr(self, "encoder_output"):
                delattr(self, "encoder_output")
            self.init_width = self.root.winfo_width()
            self.init_height = self.root.winfo_height()
            print("window resized")

            self.canvas_w = (self.init_width -300) //2
            self.canvas_h = self.init_height - 40
            self.canvas.config(width=self.canvas_w, height=self.canvas_h)
            self.canvas2.config(width=self.canvas_w, height=self.canvas_h)
            
            # this is from setup_image_display but without clearing the working image
            self.lowest_zoom_factor = min(self.canvas_w / self.original_image.width, self.canvas_h / self.original_image.height)
            self.zoom_factor = self.lowest_zoom_factor
            self.view_x = 0
            self.view_y = 0
            self.min_zoom=True
            # make a checkerboard that is always larger than preview window
            # because of rounding error
            self.checkerboard = self.create_checkerboard(self.canvas_w * 2, self.canvas_h * 2, square_size = 10)

            self.update_zoomed_view()

      

    
    def setup_image_display(self):

        self.lowest_zoom_factor = min(self.canvas_w / self.original_image.width, self.canvas_h / self.original_image.height)
  
        self.working_image = Image.new("RGBA", self.original_image.size, (0, 0, 0, 0))
        self.old_working_image = self.working_image.copy()

        self.zoom_factor = self.lowest_zoom_factor
        self.view_x = 0
        self.view_y = 0
        self.min_zoom=True
      
        # make a checkerboard that is always larger than preview window
        # because of rounding error
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
        
        self.root.minsize(width=800, height=600)  

        # Generated by pybubu designer. 
        self.main_frame = tk.Frame(self.root, container=False, name="main_frame")
        self.main_frame.configure(height=200, width=200)

        self.editor_frame = ttk.Frame(self.main_frame, name="editor_frame")
        self.editor_frame.configure(height=200, width=200)
        self.input_frame = tk.Frame(self.editor_frame, name="input_frame")
        self.input_frame.configure(height=10, width=10)
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
        self.output_frame.configure(height=10, width=10)
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
        self.openimageclipboard = ttk.Frame(
            self.OpenImage, name="openimageclipboard")
        self.openimageclipboard.configure(width=200)
        self.OpnImg = ttk.Button(self.openimageclipboard, name="opnimg")
        self.OpnImg.configure(text='Open Image')
        self.OpnImg.pack(side="left")
        self.OpnImg.configure(command=self.load_image)
        self.OpnClp = ttk.Button(self.openimageclipboard, name="opnclp")
        self.OpnClp.configure(text='Open Clipboard')
        self.OpnClp.pack(side="left")
        self.OpnClp.configure(command=self.loadclipboard)
        self.openimageclipboard.pack(side="top")
        self.EditImage = ttk.Button(self.OpenImage, name="editimage")
        self.EditImage.configure(text='Edit Image')
        self.EditImage.pack(side="top")
        self.EditImage.configure(command=self.edit_image)
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
            values='mobile_sam',
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
            values='rmbg1_4',
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
        self.Remove.configure(text='Remove mask')
        self.Remove.pack(expand=True, side="left")
        self.Remove.configure(command=self.remove_from_working_image)
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
            values='Transparent White Black Red Blue Orange Yellow Green',
            width=16)
        self.bg_color.pack(expand=False, side="right")
        self.BgSel.pack(fill="x", padx=2, pady=2, side="top")
        self.ManPaint = tk.Checkbutton(self.Options, name="manpaint")
        self.paint_mode = tk.BooleanVar()
        self.ManPaint.configure(
            text='Manual Paintbrush',
            variable=self.paint_mode)
        self.ManPaint.pack(fill="x", side="left")
        self.ManPaint.configure(command=self.paint_mode_toggle)
        self.PostMask = tk.Checkbutton(self.Options, name="postmask")
        self.ppm_var = tk.BooleanVar()
        self.PostMask.configure(
            text='Post Process Mask',
            variable=self.ppm_var)
        self.PostMask.pack(fill="x", side="top")
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
        self.Controls.pack(expand=True, fill="both", side="right")
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


        self.bg_color.bind("<<ComboboxSelected>>", lambda event: self.update_output_preview())
        self.bg_color.current(0)
        self.sam_combo.current(0)
        self.whole_image_combo.current(0)
        self.whole_image_button.configure(command = lambda: self.run_whole_image_model(None))

        # will also match quantised versions .quant
        sam_models = [
            "mobile_sam",
            "sam_vit_b_01ec64",
            "sam_vit_h_4b8939",
            "sam_vit_l_0b3195",
            ]

        whole_models = [
                "rmbg1_4",
                "rmbg1_4-quant",
                "isnet-general-use",
                "isnet-anime",
                "u2net",
                "BiRefNet", # all birefnet variations
        ]

        matches = []

        for partial_name in sam_models:
            for filename in os.listdir('Models/'):
                filename = filename.replace(".encoder.onnx","").replace(".decoder.onnx","")
                if partial_name in filename:
                    matches.append(filename)

        if len(matches) == 0:
            messagebox.showerror("No segment anything models found in Models folder")

        models = " ".join(list(dict.fromkeys(matches)))
        print("SAM models found:", models)
        self.sam_combo.configure(values=models)

        matches = []

        for partial_name in whole_models:
            for filename in os.listdir('Models/'):
                if partial_name in filename and ".onnx" in filename:
                    matches.append(filename.replace(".onnx",""))

        if len(matches) == 0:
            messagebox.showerror("No whole-image models found in Models folder")

        models = " ".join(list(dict.fromkeys(matches)))
        print("Whole image models found:", models)
        self.whole_image_combo.configure(values=models)


        
    def set_keybindings(self):

        for canvas in [self.canvas, self.canvas2]:
            canvas.bind("<ButtonPress-1>", self.start_box)
            canvas.bind("<B1-Motion>", self.draw_box)
            canvas.bind("<ButtonRelease-1>", self.end_box)
            
            canvas.bind("<Button-3>", self.generate_sam_overlay) # Negative point
            canvas.bind("<Button-4>", self.zoom)  # Linux scroll up
            canvas.bind("<Button-5>", self.zoom)  # Linux scroll down
            canvas.bind("<MouseWheel>", self.zoom) #windows scroll
            canvas.bind("<ButtonPress-2>", self.start_pan)
            canvas.bind("<B2-Motion>", self.pan)
            canvas.bind("<ButtonRelease-2>", self.end_pan)
        
        self.root.bind("<c>", lambda event: self.clear_coord_overlay())
        self.root.bind("<d>", lambda event: self.copy_entire_image())
        self.root.bind("<r>", lambda event: self.reset_all())
        self.root.bind("<a>", lambda event: self.add_to_working_image())
        self.root.bind("<z>", lambda event: self.remove_from_working_image())
        self.root.bind("<w>", lambda event: self.clear_working_image())
        self.root.bind("<s>", lambda event: self.save_as_image())
        self.root.bind("<j>", lambda event: self.quick_save_jpeg())
        self.root.bind("<p>", self.paint_mode_toggle)
        self.root.bind("<v>", lambda event: self.clear_visible_area())
        self.root.bind("<e>", lambda event: self.edit_image())
        self.root.bind("<u>", lambda event: self.run_whole_image_model("u2net", target_size=320))
        self.root.bind("<i>", lambda event: self.run_whole_image_model("isnet-general-use"))
        self.root.bind("<o>", lambda event: self.run_whole_image_model("rmbg1_4"))
        self.root.bind("<b>", lambda event: self.run_whole_image_model("BiRefNet-general-bb_swin_v1_tiny-epoch_232"))
        self.root.bind("<n>", lambda event: self.run_whole_image_model("BiRefNet-DIS-bb_pvt_v2_b0-epoch_590"))
        self.root.bind("<m>", lambda event: self.run_whole_image_model("BiRefNet-general-bb_swin_v1_tiny-epoch_232_FP16"))        
        self.root.bind("<q>", lambda event: self.undo())
        self.root.bind("<Left>", self.pan_left)
        self.root.bind("<Right>", self.pan_right)
        self.root.bind("<Up>", self.pan_up)
        self.root.bind("<Down>", self.pan_down)
        
    
    def pan_left(self, event):
        print("Panning left")
        self.view_x = max(0, self.view_x - self.pan_step)
        self.update_zoomed_view()

    def pan_right(self, event):
        print("Panning right")
        self.view_x = min(self.original_image.width - self.canvas_w / self.zoom_factor, self.view_x + self.pan_step)
        self.update_zoomed_view()

    def pan_up(self, event):
        print("Panning up")
        self.view_y = max(0, self.view_y - self.pan_step)
        self.update_zoomed_view()

    def pan_down(self, event):
        print("Panning down")
        self.view_y = min(self.original_image.height - self.canvas_h / self.zoom_factor, self.view_y + self.pan_step)
        self.update_zoomed_view()

    
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
        
        
        
        if self.min_zoom==False: 
            self.update_zoomed_view()
            if hasattr(self, "encoder_output"):
                delattr(self, "encoder_output")
            self.clear_coord_overlay()

        if self.lowest_zoom_factor == self.zoom_factor: self.min_zoom=True
    

    def start_pan(self, event):
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def pan(self, event):
        
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
            self.mask = None
            
            
            self.update_zoomed_view()
            
    
    def end_pan(self, event):
        self.panning = False    
        self.update_zoomed_view()

    
    def update_zoomed_view(self):
        
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")

        # Calculate the size of the visible area in the original image coordinates
        view_width = self.canvas_w / self.zoom_factor
        view_height = self.canvas_h / self.zoom_factor

        # Crop the visible area from the original image
        self.orig_image_crop = self.original_image.crop((
            int(self.view_x), 
            int(self.view_y),
            int(self.view_x + view_width), 
            int(self.view_y + view_height)
        ))
        
        # Resize the cropped area to fit the canvas
        if self.zoom_factor > 1 or self.panning == True:
            displayed_image = self.orig_image_crop.resize((self.canvas_w, self.canvas_h), Image.NEAREST)
        
        else:
            # Use a nicer downsampler to avoid moire in the preview window
            displayed_image = self.orig_image_crop.resize((self.canvas_w, self.canvas_h), Image.BOX)
            
        # If input image has lots of alpha this may be needed for performance reasons when panning.
        # checkerboard = self.create_checkerboard(displayed_image.width, displayed_image.height, 10)
        # displayed_image = Image.alpha_composite(checkerboard, displayed_image)

        self.orig_img_preview_w = int(self.original_image.width * self.zoom_factor)
        self.orig_img_preview_h = int(self.original_image.height * self.zoom_factor)

        # Remove the expanded area from the previous crop
        displayed_image = displayed_image.crop((0,0,
                                                min(self.canvas_w, self.orig_img_preview_w),
                                                min(self.canvas_h,self.orig_img_preview_h)
                                                ))

        self.pad_x = max(0, (self.canvas_w - self.orig_img_preview_w) // 2)
        self.pad_y = max(0, (self.canvas_h - self.orig_img_preview_h) // 2)

        
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(displayed_image)
        self.canvas.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.tk_image)


        self.update_output_preview()
        
    
    def update_output_preview(self):
        
        # Calculate the size of the visible area in the original image coordinates
        view_width = self.canvas_w / self.zoom_factor
        view_height = self.canvas_h / self.zoom_factor

        preview = self.working_image.crop((
            int(self.view_x), int(self.view_y),
            int(self.view_x + view_width),
            int(self.view_y + view_height)
        ))

                
        if self.zoom_factor > 1 or self.panning == True:

            preview = preview.resize((self.canvas_w, self.canvas_h), Image.NEAREST)
        else:
            # Nicer downsample to avoid moire
            preview = preview.resize((self.canvas_w, self.canvas_h), Image.BOX)
        

        crop_width = min(self.canvas_w, 
                            self.working_image.width * self.zoom_factor)
        crop_height = min(self.canvas_h, 
                            self.working_image.height * self.zoom_factor)

        if not self.bg_color.get() == "Transparent":
            # crop away the overcropped (padded) bit at low zoom levels
            preview = self.apply_background_color(preview.crop((0,0,
                                                        min(self.canvas_w, self.orig_img_preview_w),
                                                        min(self.canvas_h,self.orig_img_preview_h)
                                                        )), 
                                                    self.bg_color.get())
        else:
            # Composite a checkerboard to improve performance. imagetk.photoimage struggles with alpha
            # Crop checkerboard to either canvas size or preview image, if smaller
            checkerboard = self.checkerboard.crop((0,0,crop_width, crop_height))
            # crop (expand) to fit canvas size)
            checkerboard = checkerboard.crop((0,0,preview.width,preview.height))
            
            preview = Image.alpha_composite(checkerboard, preview)
       
        self.outputpreviewtk = ImageTk.PhotoImage(preview)
        self.canvas2.create_image(self.pad_x, self.pad_y, anchor=tk.NW, image=self.outputpreviewtk)    

    
    def clear_working_image(self):

        self.canvas2.delete(self.outputpreviewtk)
        self.working_image = Image.new(mode="RGBA",size=(self.original_image.width, self.original_image.height)) 
        self.update_output_preview()
    
    def reset_all(self):
        
        self.old_working_image = self.working_image
        
        self.coordinates=[]
        self.labels=[]

        self.clear_coord_overlay()

        
        self.canvas.delete(self.dots)
        if hasattr(self, 'overlay_item'):
            self.canvas.delete(self.overlay_item)
        
        if hasattr(self, "encoder_output"):
            delattr(self, "encoder_output")
            
        self.canvas2.delete(self.outputpreviewtk)
        self.working_image = Image.new(mode="RGBA",size=(self.original_image.width, self.original_image.height)) 
        self.update_zoomed_view()
        
    def undo(self):
        
        self.old_working_image, self.working_image = self.working_image, self.old_working_image
        
        self.update_output_preview()
    
    def copy_entire_image(self):
        self.old_working_image = self.working_image.copy()
        self.working_image=self.original_image.convert(mode="RGBA")
        self.update_output_preview()
    
 
    def add_to_working_image(self):

        self.old_working_image = self.working_image.copy()
        
        if self.paint_mode.get():
            mask = self.generate_paint_mode_mask()

        else:
            mask = self.mask

        full_mask = self.apply_zoomed_mask_to_full_image(mask)

        full_output = self.original_image.convert("RGBA")
        full_output.putalpha(full_mask)
        
        self.working_image = Image.alpha_composite(self.working_image, full_output)

        self.update_output_preview()    
        
    def remove_from_working_image(self):
        
        self.old_working_image = self.working_image.copy()
        
        if self.paint_mode.get():
            mask = self.generate_paint_mode_mask()
            
        else:
            mask = self.mask

        full_mask = self.apply_zoomed_mask_to_full_image(mask)
        
        empty = Image.new("RGBA", self.original_image.size, 0)
        self.working_image = Image.composite(self.working_image, empty, ImageOps.invert(full_mask))

        self.update_output_preview()
    
    def clear_visible_area(self):
        mask_old = self.mask.copy()
        self.mask = Image.new("L", self.orig_image_crop.size, 255)
        self.remove_from_working_image()
        self.mask = mask_old
    
    def apply_zoomed_mask_to_full_image(self, zoomed_mask):
        full_mask = Image.new('L', self.original_image.size, 0)
        paste_box = (
            int(self.view_x),
            int(self.view_y),
            int(self.view_x + self.orig_image_crop.width),
            int(self.view_y + self.orig_image_crop.height)
        )
        
        full_mask.paste(zoomed_mask, paste_box)
        return full_mask

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
        
        self.mask = Image.new("L",self.orig_image_crop.size,0)
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
          
        if dx < self.min_rect_size and dy < self.min_rect_size:
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
            self.generate_sam_overlay(event)
    
    def box_event(self, scaled_coords):
        self._initialise_sam_model()
        
        self.clear_coord_overlay()

        self.coordinates = [[scaled_coords[0], scaled_coords[1]], [scaled_coords[2], scaled_coords[3]]]
        self.labels = [2, 3]  # Assuming 2 for top-left and 3 for bottom-right
        
        self.mask = self.sam_calculate_mask(self.orig_image_crop, self.sam_encoder, self.sam_decoder, self.coordinates, self.labels)
        
        self.generate_coloured_overlay()

        self.coordinates = []
        self.labels =[]
    
    def load_whole_image_model(self, model_name):
        
        if not hasattr(self, f"{model_name}_session"):
            
            self.status_label.config(text=f"Loading {model_name}", fg=STATUS_PROCESSING)
            self.status_label.update()

            setattr(self, f"{model_name}_session", ort.InferenceSession(f'{MODEL_ROOT}{model_name}.onnx'))
        return getattr(self, f"{model_name}_session")
    
    
    
    def run_whole_image_model(self, model_name, target_size = 1024):
        
        # if button has been clicked
        if model_name == None:
            #selected = self.whole_model_choice.get()
            #model_name = self.model_map.get(selected)
            model_name = self.whole_image_combo.get()
            target_size = 320 if model_name == "u2net" else 1024

        try: 
            session = self.load_whole_image_model(model_name)
        except Exception as e:
            print(e)
            self.status_label.config(text=f"ERROR: {e}", fg=STATUS_PROCESSING)
            self.root.update()
            messagebox.showerror("Error", e)
            return


        self.status_label.config(text=f"Processing {model_name}", fg=STATUS_PROCESSING)
        self.status_label.update()

        self.mask = self.generate_whole_image_model_mask(self.orig_image_crop, session, self.ppm_var.get(), target_size)
        
        self.generate_coloured_overlay()            
     
        
    def generate_whole_image_model_mask(self, image,  session, ppm=False, target_size=1024):
        
        def sigmoid(mat):
            # For BiRefNet
            return 1/(1+np.exp(-mat))   
        # Preprocess
        
        # Resize and normalise the image
        # Can't remember where I found this...
        # input_image = image.resize((target_size, target_size), Image.BICUBIC)
        # input_image = np.array(input_image).astype(np.float32)
        # input_image = input_image / 255.0
        # input_image = np.transpose(input_image, (2, 0, 1))  # Change data layout from HWC to CHW
        # input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        

        #taken from REMBG. seems to make a slightly less transparent mask
        input_image = image.convert("RGB").resize((target_size,target_size), Image.BICUBIC)
        
        mean = (0.485, 0.456, 0.406)
        std = (1.0, 1.0, 1.0)

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
        
        if "BiRefNet" in session._model_path:
        
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
        

        if ppm:
            #Apply morphological operations to smooth edges
            mask = mask.point(lambda p: p > 128 and 255)  # Binarize the mask
            mask_array = np.array(mask)
            mask_array = binary_dilation(mask_array, iterations=1)
            mask_array = binary_erosion(mask_array, iterations=1)
            mask = Image.fromarray(mask_array.astype(np.uint8) * 255)
        
            # Apply Gaussian blur to further smooth the edges
            #mask = mask.filter(ImageFilter.GaussianBlur(3))

        
        mask = mask.convert("L")
        
        return mask


    

      
    def generate_coloured_overlay(self):
            
        self.overlay = ImageOps.colorize(self.orig_image_crop.convert("L"), black="blue", white="white") 
        self.overlay.putalpha(self.mask) 
    
        
        self.scaled_overlay = self.overlay.resize((self.canvas_w, self.canvas_h), Image.NEAREST)
        
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
        
        
        if self.paint_mode.get():

            self.clear_coord_overlay()
            
            for canvas in [self.canvas, self.canvas2]:
                
                canvas.config(cursor="circle")

                canvas.bind("<ButtonPress-1>", self.paint_draw_point)
                canvas.bind("<B1-Motion>", self.paint_draw_line)
                canvas.bind("<ButtonRelease-1>", self.paint_reset_coords)
            
        else:    
#      
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
        if not hasattr(self, "sam_encoder"):
            self.status_label.config(text="Loading model", fg=STATUS_PROCESSING)
            self.status_label.update()
            self.sam_model = MODEL_ROOT + self.sam_model_choice.get()
            self.sam_encoder = ort.InferenceSession(self.sam_model + ".encoder.onnx")
            self.sam_decoder = ort.InferenceSession(self.sam_model + ".decoder.onnx")
        elif not self.sam_model == MODEL_ROOT + self.sam_model_choice.get():
            self.status_label.config(text="Loading model " + self.sam_model_choice.get(), fg=STATUS_PROCESSING)
            self.status_label.update()
            self.sam_model = MODEL_ROOT + self.sam_model_choice.get()
            self.sam_encoder = ort.InferenceSession(self.sam_model + ".encoder.onnx")
            self.sam_decoder = ort.InferenceSession(self.sam_model + ".decoder.onnx")
            self.clear_coord_overlay()
            if hasattr(self, "encoder_output"): delattr(self, "encoder_output")


    def generate_sam_overlay(self, event):

        self._initialise_sam_model()
    
        x, y = (event.x-self.pad_x) / self.zoom_factor, (event.y-self.pad_y) / self.zoom_factor
        self.coordinates.append([x, y])
        
        self.labels.append(event.num if event.num <= 1 else 0)
        
        # Draw dot so user can see responsiveness,
        # as model might take a while to run.
        self.draw_dot(event.x, event.y, event.num)
        self.canvas.update()
        
        self.mask = self.sam_calculate_mask(self.orig_image_crop, 
                                            self.sam_encoder, self.sam_decoder, self.coordinates, self.labels)
        self.generate_coloured_overlay()
        
        # Repeated to ensure the dot stays on top
        self.draw_dot(event.x, event.y, event.num)

    def sam_calculate_mask(self,
        img,
        sam_encoder,
        sam_decoder,
        coordinates,
        labels,
        ):
        """
        Predict masks for an input image.

        Returns:
            List[PILImage]: A list of masks generated by the decoder.
        """
       

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
            self.root.update()
            
            start = timer()
            self.encoder_output = sam_encoder.run(None, encoder_inputs)

            status = f"{round(timer()-start,2)} seconds to calculate embedding | "

            self.status_label.config(text=status, fg=STATUS_NORMAL)
            
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
        
        print(user_filename)

        workimg = self.apply_background_color(self.working_image, "White")
        workimg = workimg.convert("RGB")
        if self.image_exif:
            workimg.save(user_filename, quality=90, exif=self.image_exif)
        else:
            workimg.save(user_filename, quality=90)
        print("Saved to "+ user_filename)
        self.status_label.config(text="Saved to "+ user_filename)
        self.canvas.update()

    
    def show_save_options(self):
        option_window = tk.Toplevel(self.root)
        option_window.title("Save Options")
        option_window.geometry("300x280")  
        option_window.resizable(False, False)
        
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
        
        update_quality_state() 
        
        result = {"file_type": None, "quality": None}
        
        def on_ok():
            result["file_type"] = file_type.get()
            result["quality"] = quality.get()
            self.save_file_type = file_type.get()
            self.save_file_quality = quality.get()
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
        
        initial_file = os.path.splitext(os.path.basename(self.save_file))[0] + ext
        user_filename = asksaveasfilename(
            title="Save as",
            defaultextension=ext,
            filetypes=[(file_type, "*" + ext)],
            initialdir=os.path.dirname(self.save_file),
            initialfile=initial_file
        )

        if not user_filename:
            return

        if not user_filename.lower().endswith(ext):
            user_filename += ext

        if not self.bg_color.get() == "Transparent":
            workimg = self.apply_background_color(self.working_image, self.bg_color.get())
        else:
            workimg = self.working_image

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

        self.canvas.update()

    

    def update_bg_color(self):
                
        if self.bg_color_var.get() == 1:
            self.bgcolor = (255, 255, 255, 255)

        else:
            self.bgcolor = None
            
        self.update_output_preview()


    def apply_background_color(self, img, color):

        #r, g, b, a = color
        colored_image = Image.new("RGBA", img.size, color)
        colored_image.paste(img, mask=img)

        return colored_image
    
    def on_closing(self):
        
        print("closing")
        self.root.destroy()
        
    def edit_image(self):
        if messagebox.askyesno("Continue?", "Editing the original image will reset the current working output image. Would you like to edit the image?"): 

            editor = ImageEditor(self.root, self.original_image, self.file_count)
            self.root.wait_window(editor.crop_window)
            if editor.final_image:
                self.original_image = editor.final_image
                self.initialise_new_image()
        
    def loadclipboard(self):
        
        img = ImageGrab.grabclipboard()
        
        
        if "PIL." in str(type(img)):
            self.original_image = img
            
            
            self.initialise_new_image()
            self.save_file = None
            self.save_file_jpg = None
            
            file_path = asksaveasfilename(
                title="Image File Name (no file extension)",
            )
            self.save_file = file_path+"_nobg.png"
            print(self.save_file)
            self.status_label.config(text=f"File will be saved to {self.save_file}")
            self.save_file_jpg = file_path+"_nobg.jpg"
            
        else:
            messagebox.showerror("Error", f"No image found on clipboard.\nClipboard contains {type(img)}")
        
    def load_image(self):
        image_path = askopenfilename(
            title="Select an Image",
            filetypes=pillow_formats
        )
        self.original_image = Image.open(image_path)
        self.original_image = ImageOps.exif_transpose(self.original_image)
        try:
            self.image_exif = self.original_image.info['exif']
            print("EXIF data found!")
        except KeyError:
            self.image_exif = None
            print("No EXIF data found.")
        
        self.initialise_new_image()
        
    def initialise_new_image(self):
        

        self.canvas2.delete("all")
        
        self.setup_image_display()
        self.update_zoomed_view()
        self.clear_coord_overlay()
        self.reset_all()
        
    def show_help(self):
        message = """
            Interactive Background Remover by Sean (Prickly Gorse)

A user interface for removing backgrounds using interactive models (Segment Anything) and Whole Image Models (u2net, disnet, rmbg, BirefNet)

Load your image, and either run one of the whole image models (u2net <u>, disnet <i>, rmbg <o>, BiRefNet) or click/draw a box to run Segment Anything. Left click is a positive point, right is a negative (avoid this area) point.

The original image is displayed on the left, the current image you are working on is displayed to the right.

Type A to add the current mask to your image, Z to remove.

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
<z> Remove current mask from working image
<q> Undo last action
<p> Manual paintbrush mode
<c> Clear current mask (and coordinate points)
<w> Reset the current working image
<r> Reset everything (image, masks, coordinates)
<v> Clear the visible area on the working image
<s> Save as....
<j> Quick save JPG with white background

Whole image models (if downloaded to Models folder)
<u> u2net
<i> disnet
<o> rmbg
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

        screen_width = (m.width -250)
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
        

        self.canvas_crop_window = Canvas(self.crop_window, width=screen_width, height=self.scaled_height)
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
            'brightness': (0.1, 2.0, 1.0),
            'contrast': (0.1, 2.0, 1.0),
            'saturation': (0.1, 2.0, 1.0),
            'steepness': (0.01, 0.5, 0.1),
            'white_balance': (2000, 10000, 6500),
            'unsharp_radius': (0.1, 50, 1.0),
            'unsharp_amount': (0, 5, 0),
            'unsharp_threshold': (0, 10, 3)
        }
        
        
        
        for param, (min_val, max_val, default) in slider_params.items():
            self.sliders[param] = tk.Scale(
                self.slider_frame,
                from_=min_val,
                to=max_val,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                label=param.capitalize(),
                length=300,
                width=20
            )
            # this command triggers the function to run
            self.sliders[param].set(default)
            self.sliders[param].pack(pady=0)
        
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
        self.common_adjustments = Button(self.reset_frame, text="Common Adjustments", command=self.common_slider_adjustment)
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
         
    def apply_unsharp_mask(self, image, radius, amount, threshold):
        if amount == 0:
            return image
        
        img_array = np.array(image).astype(np.float32) / 255.0
        
        blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
        
        unsharp_mask = img_array - blurred
        
        threshold = threshold / 255.0
        mask = np.abs(unsharp_mask) > threshold
        img_sharpened = img_array + amount * unsharp_mask * mask
        
        img_sharpened = np.clip(img_sharpened * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_sharpened) 
    
    def reset_sliders(self):
         default_values = {
             'highlight': 1.0,
             'midtone': 1.0,
             'shadow': 1.0,
             'brightness': 1.0,
             'contrast': 1.0,
             'saturation': 1.0,
             'steepness': 0.1,
             'white_balance': 6500,
             'unsharp_radius': 1.0,
             'unsharp_amount': 0,
             'unsharp_threshold': 3
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
             'steepness': 0.1,
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
        screen_width = self.crop_window.winfo_width() - 300
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
                            saturation, steepness, white_balance,
                            unsharp_radius, unsharp_amount, unsharp_threshold):

         img_array = np.array(image)

         # Combine all masks into a single operation
         x = np.arange(256, dtype=np.float32)
         highlight_mask = self.smooth_transition(x, 192, steepness)
         shadow_mask = 1 - self.smooth_transition(x, 64, steepness)
         midtone_mask = 1 - highlight_mask - shadow_mask
     
         # Create a lookup table
         lut = (x * highlight * highlight_mask +
                x * midtone * midtone_mask +
                x * shadow * shadow_mask).clip(0, 255).astype(np.uint8)
     
         adjusted = lut[img_array]
     
         adjusted_image = Image.fromarray(adjusted)
         
         # Combine enhancement operations
         if saturation != 1.0:
             enhancer = ImageEnhance.Color(adjusted_image)
             adjusted_image = enhancer.enhance(saturation)
         if brightness != 1.0:
             enhancer = ImageEnhance.Brightness(adjusted_image)
             adjusted_image = enhancer.enhance(brightness)
         if contrast != 1.0:
             enhancer = ImageEnhance.Contrast(adjusted_image)
             adjusted_image = enhancer.enhance(contrast)
         if white_balance != 6500:
            adjusted_image = self.adjust_white_balance(adjusted_image, white_balance)
            
         if unsharp_amount > 0:
            adjusted_image = self.apply_unsharp_mask(adjusted_image, unsharp_radius, unsharp_amount, unsharp_threshold)
     
         return adjusted_image
    
    def adjust_white_balance(self, image, temperature):
        rgb = self.kelvin_to_rgb(temperature)
        r_factor, g_factor, b_factor = [x / max(rgb) for x in rgb]
    
        img_array = np.array(image)
    
        avg_brightness = np.mean(img_array[:,:,:3])
    
        img_array[:,:,0] = np.clip(img_array[:,:,0] * r_factor, 0, 255)
        img_array[:,:,1] = np.clip(img_array[:,:,1] * g_factor, 0, 255)
        img_array[:,:,2] = np.clip(img_array[:,:,2] * b_factor, 0, 255)
    
        new_avg_brightness = np.mean(img_array[:,:,:3])
    
        brightness_factor = avg_brightness / new_avg_brightness
        img_array[:,:,:3] = np.clip(img_array[:,:,:3] * brightness_factor, 0, 255)
    
        adjusted = Image.fromarray(img_array.astype('uint8'), mode=image.mode)
    
        return adjusted


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

    if len(sys.argv) <= 1:
        print("No arguments were given")
        
        pillow_formats = [
            ("All Image Files", "*.bmp *.gif *.jpg *.jpeg *.png *.tif *.tiff *.webp"),
            ("BMP", "*.bmp"),
            ("GIF", "*.gif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("TIFF", "*.tif *.tiff"),
            ("WEBP", "*.webp"),
            ("All Files", "*.*")
        ]
        
        file_path = askopenfilename(
            title="Select an Image",
            filetypes=pillow_formats
        )


        root = tk.Tk()
        app = ImageClickApp(root, file_path)
        root.mainloop()
    else:
        files = sys.argv[1:]
        for count, file_path in enumerate(files):
            #file_path = sys.argv[1]   
            root = tk.Tk()
            app = ImageClickApp(root, file_path, f' - Image {count+1} of {len(files)}')
            root.mainloop()
    