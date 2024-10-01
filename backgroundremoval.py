
#!/usr/bin/env python3
from line_profiler import profile
import tkinter as tk
from tkinter import Checkbutton, Frame, Button, OptionMenu, messagebox
from tkinter import Canvas, IntVar, StringVar, BooleanVar
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageEnhance, ImageGrab
#from PIL.ExifTags import TAGS, GPSTAGS
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
        self.coordinates = []
        self.labels=[]
        self.bgcolor = None
        self.dots=[]
        
        self.min_rect_size = 5
        
                
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.last_x, self.last_y = 0, 0
        

        
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
        
        
    
        m = [m for m in get_monitors() if m.is_primary][0]
        
        
        self.canvas_w = (m.width -250) //2
        self.canvas_h = m.height-100 

        
        if platform.system() == "Windows":
            self.root.state('zoomed')
        elif platform.system() == "Darwin":
            self.root.state('zoomed') #unsure if works
        else:
            self.root.attributes('-zoomed', True)

 
        self.setup_image_display()
        self.build_gui()
        
        self.update_zoomed_view()


        self.mask = Image.new("L", (int(self.orig_image_crop.width), 
                                            int(self.orig_image_crop.height)),0)

        self.set_keybindings()


        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        

    

    
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

        self.imgFrame = Frame(root)

        self.canvas = Canvas(self.imgFrame, width=self.canvas_w, height=self.canvas_h,
                             highlightthickness=1, highlightbackground="black")
        self.canvas.pack(side=tk.LEFT)

        
        self.canvas2 = Canvas(self.imgFrame, width=self.canvas_w, height=self.canvas_h,
                              highlightthickness=1, highlightbackground="black")
        self.canvas2.pack(side=tk.LEFT)

        
        self.imgFrame.pack(side=tk.LEFT)
        
        self.checkbox_frame = Frame(root)
        self.checkbox_frame.pack()
        
        self.bg_color_var = IntVar()
        self.ppm_var = BooleanVar()
        self.paint_mode = BooleanVar()
        
        self.load_frame= Frame(self.checkbox_frame)
        
        self.load_image_button = Button(self.load_frame, text="Open File", command=self.load_image )
        self.load_image_button.pack(side=tk.LEFT)
        
        self.loadfromclipboardbutton = Button(self.load_frame, text="Open Clipboard", command=self.loadclipboard )
        self.loadfromclipboardbutton.pack(side=tk.LEFT)
        
        self.load_frame.pack()
        
        
        
        
        self.preprocess_image = Button(self.checkbox_frame, text="EDIT IMAGE (clears all) <e>", command=self.edit_image )
        self.preprocess_image.pack(pady=(10,20))


        


        
        
        
        
        self.clearbutton2 = Button(self.checkbox_frame, text="Clear points and overlay <c>", command=self.clear_coord_overlay)
        
        self.clearbutton2.pack()
        
        self.clearbutton3 = Button(self.checkbox_frame, text="Reset Output Image <w>", command=self.clear_working_image)
        self.clearbutton3.pack()
        
        self.clearbutton = Button(self.checkbox_frame, text="Reset everything <r>", command=self.reset_all)
        self.clearbutton.pack()
        
        
        self.whole_image_label = tk.Label(self.checkbox_frame, text="Whole-image (preview area) models")
        self.whole_image_label.pack(pady=(20,0))
        
        self.whole_model_choice = StringVar(self.checkbox_frame)
        self.whole_model_choice.set("rmbg1_4 <o>") # default value

        self.model_map = {
            "u2net <u>": "u2net",
            "isnet-general-use <i>": "isnet-general-use",
            "rmbg1_4 <o>": "rmbg1_4",
            "BiRefNet-general_swin_tiny <b>": "BiRefNet-general-bb_swin_v1_tiny-epoch_232",
            "BiRefNet-DIS-bb_pvt": "BiRefNet-DIS-bb_pvt_v2_b0-epoch_590",
            "BiRefNet-general_swin_tiny_FP16": "iRefNet-general-bb_swin_v1_tiny-epoch_232_FP16",
        }
    
        self.image_model = OptionMenu(self.checkbox_frame, self.whole_model_choice, 
                                       *self.model_map.keys())
        self.image_model.pack(pady=(0,0))
        
        self.ppm_check = Checkbutton(self.checkbox_frame, text="Post Process Mask", variable=self.ppm_var)
        self.ppm_check.pack()
        
        self.whole_button = Button(self.checkbox_frame, text="Run model", command=lambda: self.run_unet_model(None))
        self.whole_button.pack(pady=(0,20))

        
        self.sam_label = tk.Label(self.checkbox_frame, text="Segment Anything Model:")
        self.sam_label.pack()
        
        self.sam_model_choice = StringVar(self.checkbox_frame)
        self.sam_model_choice.set("mobile_sam") # default value

        self.sam_dropdown = OptionMenu(self.checkbox_frame, self.sam_model_choice, 
                                       "mobile_sam", "sam_vit_b_01ec64", 
                                       "sam_vit_b_01ec64.quant", "sam_vit_l_0b3195.quant", "sam_vit_h_4b8939.quant", "sam2_hiera_tiny")
        self.sam_dropdown.pack(pady=(0,0))    
        
        
        
        self.undobutton = Button(self.checkbox_frame, text="Undo <q>", command=self.undo)
        self.undobutton.pack(pady=(20,0))
        
        self.appendbutton = Button(self.checkbox_frame, text="Add to Image <a>", command=self.add_to_working_image)
        self.appendbutton.pack()
        
        self.removebutton = Button(self.checkbox_frame, text="Remove from Image <z>", command=self.remove_from_working_image)
        self.removebutton.pack()
        
        self.copy_button = Button(self.checkbox_frame, text="Copy original to canvas <d>", command=self.copy_entire_image)
        self.copy_button.pack()
        
        self.clear_visible_button = Button(self.checkbox_frame, text="Clear visible area <v>", command=self.clear_visible_area)
        self.clear_visible_button.pack()
        
        self.bg_color_check = Checkbutton(self.checkbox_frame, text="White Background", variable=self.bg_color_var, command=self.update_bg_color)
        self.bg_color_check.pack(pady=(20,0))
        
        self.paintmodebutton = Checkbutton(self.checkbox_frame, text="Manual Paint <p>", variable=self.paint_mode, command=self.paint_mode_toggle)
        self.paintmodebutton.pack(pady=(20,0))
        
        
        
        

        self.savebutton = Button(self.checkbox_frame, text="Save PNG <s>", command=self.save_as_png)
        self.savebutton.pack(pady=(20,0))
        
        self.savebuttonjpg = Button(self.checkbox_frame, text="Save JPG white background <j>", command=self.save_as_jpeg)
        self.savebuttonjpg.pack()
        
        
        self.status_title = tk.Label(self.checkbox_frame, text="Current Status:")
        self.status_title.pack(pady=(20,0))
        self.status_label = tk.Label(self.checkbox_frame, text="", wraplength=200)
        self.status_label.pack()
        
        self.zoom_label = tk.Label(self.checkbox_frame, text="Zoom: "+ str(int(self.zoom_factor *100))+"%")
        self.zoom_label.pack(pady=(20,0))
        
        
        
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
        
        #In Python, when you pass a function as a callback (e.g., for a button click or a keypress), 
        #you should pass the function itself without parentheses, so that it is not called immediately. 
        #Instead, it should be triggered by the event.
        
        self.root.bind("<c>", lambda event: self.clear_coord_overlay())
        self.root.bind("<d>", lambda event: self.copy_entire_image())
        self.root.bind("<r>", lambda event: self.reset_all())
        self.root.bind("<a>", lambda event: self.add_to_working_image())
        self.root.bind("<z>", lambda event: self.remove_from_working_image())
        self.root.bind("<w>", lambda event: self.clear_working_image())
        self.root.bind("<s>", lambda event: self.save_as_png())
        self.root.bind("<j>", lambda event: self.save_as_jpeg())
        self.root.bind("<p>", self.paint_mode_toggle)
        self.root.bind("<v>", lambda event: self.clear_visible_area())
        self.root.bind("<e>", lambda event: self.edit_image())
        self.root.bind("<u>", lambda event: self.run_unet_model("u2net", target_size=320))
        self.root.bind("<i>", lambda event: self.run_unet_model("isnet-general-use"))
        self.root.bind("<o>", lambda event: self.run_unet_model("rmbg1_4"))
        self.root.bind("<b>", lambda event: self.run_unet_model("BiRefNet-general-bb_swin_v1_tiny-epoch_232"))
        self.root.bind("<n>", lambda event: self.run_unet_model("BiRefNet-DIS-bb_pvt_v2_b0-epoch_590"))
        self.root.bind("<m>", lambda event: self.run_unet_model("BiRefNet-general-bb_swin_v1_tiny-epoch_232_FP16"))        
        self.root.bind("<q>", lambda event: self.undo())
        
    
    
    
    def zoom(self, event):
        print(event.delta)
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

    @profile
    def update_zoomed_view(self):
        
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
        
    @profile
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

        if self.bgcolor:
            # crop away the overcropped (padded) bit at low zoom levels
            preview = self.apply_background_color(preview.crop((0,0,
                                                        min(self.canvas_w, self.orig_img_preview_w),
                                                        min(self.canvas_h,self.orig_img_preview_h)
                                                        )), 
                                                    self.bgcolor)
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
        
        self.mask = Image.new("L", self.orig_image_crop.size, 255)
        self.remove_from_working_image()
    
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
    
    def load_unet_model(self, model_name):
        
        if not hasattr(self, f"{model_name}_session"):
            self.status_label.config(text=f"Loading {model_name}", fg=STATUS_PROCESSING)
            self.status_label.update()
            setattr(self, f"{model_name}_session", ort.InferenceSession(f'{MODEL_ROOT}{model_name}.onnx'))
        return getattr(self, f"{model_name}_session")
    
    
    
    def run_unet_model(self, model_name, target_size = 1024):
        
        # if button has been clicked
        if model_name == None:
            selected = self.whole_model_choice.get()
            model_name = self.model_map.get(selected)
            target_size = 320 if model_name == "u2net" else 1024

        session = self.load_unet_model(model_name)

        self.status_label.config(text=f"Processing {model_name}", fg=STATUS_PROCESSING)
        self.canvas.update()

        self.mask = self.generate_unet_mask(self.orig_image_crop, session, self.ppm_var.get(), target_size)
        
        self.generate_coloured_overlay()            
     
        
    def generate_unet_mask(self, image,  session, ppm=False, target_size=1024):
        
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
        
        #self.inv_zoom = 1/ self.zoom_factor
        
        ellipse_radius = PAINT_BRUSH_DIAMETER/2 / self.zoom_factor
        line_width = int(PAINT_BRUSH_DIAMETER / self.zoom_factor)
        
        for x1, y1, x2, y2 in self.lines:
            
            draw.line((x1, y1, x2, y2), fill='white', width=line_width)
            draw.ellipse((x1-ellipse_radius, y1-ellipse_radius, x1+ellipse_radius, y1+ellipse_radius), fill='white')
            draw.ellipse((x2-ellipse_radius, y2-ellipse_radius, x2+ellipse_radius, y2+ellipse_radius), fill='white')
        
        return img
    
    def paint_mode_toggle(self, event=None):
        
        if event:
            self.paintmodebutton.toggle()
        
        self.root.update()
        
        
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
            self.status_label.config(text="Loading model", fg=STATUS_PROCESSING)
            self.status_label.update()
            self.sam_model = MODEL_ROOT + self.sam_model_choice.get()
            self.sam_encoder = ort.InferenceSession(self.sam_model + ".encoder.onnx")
            self.sam_decoder = ort.InferenceSession(self.sam_model + ".decoder.onnx")
            delattr(self, "encoder_output")


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
            
            
            self.status_label.config(text="Calculating Image Embedding\nMay take a while", fg=STATUS_PROCESSING)
            self.root.update()
            
            start = timer()
            self.encoder_output = sam_encoder.run(None, encoder_inputs)

            status = f"{round(timer()-start,2)} seconds embedding\n"

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
        
        
        
        status += f"{round(timer()-start,2)} seconds inference"
        status += f"\n{round(float(a[0][0]),2)} confidence score"

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

    
    def save_dialog(self, ext):

        self.status_label.config(text="", fg=STATUS_NORMAL)

        dir_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)

        file_name_nobg = os.path.splitext(file_name)[0] + "_nobg."+ext

        user_filename = asksaveasfilename(title = "Save as", 
                                          defaultextension='.'+ext, filetypes=[(ext, "."+ext)],
                                          
                                          initialdir=dir_path, initialfile=file_name_nobg,
                                          )

        if len(user_filename) == 0:
            return None
        
        if not user_filename.endswith("."+ext): user_filename+="."+ext
        
        return user_filename
    
    
    def save_as_png(self):

        user_filename = self.save_dialog("png")
        if not user_filename: return
        
        if self.bgcolor: 
            workimg = self.apply_background_color(self.working_image, self.bgcolor) 
        else: 
            workimg = self.working_image
        
        self.status_label.config(text="Saving to PNG (slow)", fg=STATUS_PROCESSING)
        self.root.update()
        if self.image_exif:
            workimg.save(user_filename, lossless=True, optimize=True, exif=self.image_exif) #lossless is for webp, optimize is for png. doesnt seem to matter if unused parameters added here
        else:
            workimg.save(user_filename, lossless=True, optimize=True)
        print("Saved to "+ user_filename)
        self.status_label.config(text="Saved to "+ user_filename, fg=STATUS_NORMAL)
        self.canvas.update()
        
    def save_as_jpeg(self):
        
        user_filename = self.save_dialog("jpg")
        if not user_filename: return

        print(user_filename)

        workimg = self.apply_background_color(self.working_image, (255, 255, 255, 255))
        workimg = workimg.convert("RGB")
        if self.image_exif:
            workimg.save(user_filename, quality=90, exif=self.image_exif)
        else:
            workimg.save(user_filename, quality=90)
        print("Saved to "+ user_filename)
        self.status_label.config(text="Saved to "+ user_filename)
        self.canvas.update()
        
    def update_bg_color(self):
                
        if self.bg_color_var.get() == 1:
            self.bgcolor = (255, 255, 255, 255)

        else:
            self.bgcolor = None
            
        self.update_output_preview()


    def apply_background_color(self, img, color):

        r, g, b, a = color
        colored_image = Image.new("RGBA", img.size, (r, g, b, a))
        colored_image.paste(img, mask=img)

        return colored_image
    
    def on_closing(self):
        
        print("closing")
        self.root.destroy()
        
    def edit_image(self):
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
                #filetypes=pillow_formats
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
        # self.crop_window.wm_attributes('-zoomed', 1)
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

        # Get window dimensions
        # screen_width = self.crop_window.winfo_width() - 300
        # screen_height = self.crop_window.winfo_height() - 100
                
        self.image_ratio = min(screen_width / self.original_image.width, screen_height / self.original_image.height)
        self.scaled_width = int(self.original_image.width * self.image_ratio)
        self.scaled_height = int(self.original_image.height * self.image_ratio)
        
        self.display_image = self.original_image.resize((self.scaled_width, self.scaled_height))
        self.crop_window_tk_image = ImageTk.PhotoImage(self.display_image)
        

        self.canvas_crop_window = Canvas(self.crop_window, width=screen_width, height=self.scaled_height)
        self.canvas_crop_window.pack(side=tk.LEFT)
        
        self.canvas_crop_window.create_image(screen_width/2, screen_height/2, image=self.crop_window_tk_image)
        
        
        self.slider_frame = tk.Frame(self.crop_window, width=300)
        self.slider_frame.pack(side=tk.LEFT)
        
        
        
        
        
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
                #add this later to stop the lag during gui initialisation
                #command=self.update_crop_preview,
                length=300,
                width=20
            )
            # this command triggers the function to run
            self.sliders[param].set(default)
            self.sliders[param].pack(pady=0)
        
        # Add the command after all sliders are created and set
        for slider in self.sliders.values():
            slider.config(command=self.update_crop_preview)
            
        # Create a frame for rotation buttons
        self.rotation_frame = Frame(self.slider_frame)
        self.rotation_frame.pack(pady=10)
           
        # Add rotation buttons
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
            ("All Image Files", "*.bmp *.dib *.dcx *.eps *.ps *.gif *.im *.jpg *.jpeg *.pcd *.pcx *.pdf *.png *.pbm *.pgm *.ppm *.psd *.tif *.tiff *.xbm *.xpm"),
            ("BMP", "*.bmp *.dib"),
            ("DCX", "*.dcx"),
            ("EPS", "*.eps *.ps"),
            ("GIF", "*.gif"),
            ("IM", "*.im"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PCD", "*.pcd"),
            ("PCX", "*.pcx"),
            ("PDF", "*.pdf"),
            ("PNG", "*.png"),
            ("PBM", "*.pbm"),
            ("PGM", "*.pgm"),
            ("PPM", "*.ppm"),
            ("PSD", "*.psd"),
            ("TIFF", "*.tif *.tiff"),
            ("XBM", "*.xbm"),
            ("XPM", "*.xpm"),
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
    