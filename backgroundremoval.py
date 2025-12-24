#!/usr/bin/env python3

try:
    import pyi_splash # type: ignore
    pyi_splash.update_text("Loading Packages")
except: pass
import argparse
import sys
import os
import math
import numpy as np
import cv2
import gc
from timeit import default_timer as timer
#from line_profiler import profile

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QSplitter, QDialog, QScrollArea, 
                             QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem, 
                             QTextEdit, QSizePolicy, QRadioButton, QButtonGroup, QInputDialog, 
                             QProgressBar, QStyle)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal
from PyQt6.QtGui import (QPixmap, QImage, QColor, QPainter, QPainterPath, QPen, QBrush,
                         QKeySequence, QShortcut, QCursor, QIcon, QPalette)

from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageGrab, ImageFilter, ImageChops
import onnxruntime as ort



try: pyi_splash.update_text("Loading App Scripts")
except: pass

import src.settings_download_manager as settings_download_manager
from src.ui_widgets import CollapsibleFrame, SynchronisedGraphicsView, ThumbnailList
from src.ui_dialogs import SaveOptionsDialog, ImageEditorDialog
from src.trimap_editor import TrimapEditorDialog
from src.utils import pil2pixmap, numpy_to_pixmap
from src.constants import PAINT_BRUSH_SCREEN_SIZE, UNDO_STEPS, SOFTEN_RADIUS

try: pyi_splash.update_text("Loading pymatting (Compiles on first run, approx 1-2 minutes)")
except: pass

print("Loading pymatting. On first run this will take a minute or two as it compiles")
from src.model_manager import ModelManager 


class InferenceWorker(QThread):
    """Super simple threaded worker that can take functions"""
    finished = pyqtSignal(object)  # Success result (mask, image, etc.)
    error = pyqtSignal(str)       # Error message

    def __init__(self, task_fn, *args, **kwargs):
        super().__init__()
        self.task_fn = task_fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BackgroundRemoverGUI(QMainWindow):
    def __init__(self, image_paths, cli_args=None):
        super().__init__()
        
        try: pyi_splash.update_text("App Initialisation")
        except: pass
        
        # Persistent settings
        self.settings = QSettings("PricklyGorse", "InteractiveBackgroundRemover")

        # Set up default model location
        if getattr(sys, 'frozen', False):
            script_base = os.path.dirname(sys.executable)
        else:
            script_base = os.path.dirname(os.path.abspath(__file__))
            
        default_model_dir = os.path.join(script_base, "Models")
        self.model_root_dir = self.settings.value("model_root_dir", default_model_dir)

        cache_root_dir = os.path.join(script_base, "Models", "cache")


        self.model_manager = ModelManager(self.model_root_dir, cache_root_dir)

        self.image_paths = image_paths if image_paths else []
        self.current_image_index = 0
        self.setWindowTitle("Interactive Image Background Remover")
        self.resize(1600, 900)

        self.coordinates = []
        self.labels = []
        self.undo_history = []
        self.redo_history = []
        self.paint_mode = False
        self.image_exif = None
        self.blur_radius = 30

        
        self.original_image = None
        self.working_image = None
        self.working_mask = None
        self.last_trimap = None

        self.model_output_mask = None
        try: pyi_splash.update_text("Loading UI")
        except: pass
        self.init_ui()

        if cli_args.binary:
            self.chk_post.setChecked(True)
        if cli_args.soften:
            self.chk_soften.setChecked(True)
        if cli_args.alpha_matting:
                self.chk_alpha_matting.setChecked(True)
        if cli_args.colour_correction:
                self.chk_estimate_foreground.setChecked(True)
        if cli_args.shadow:
                self.chk_shadow.setChecked(True)

        if cli_args.bg_colour:
            # Search for the color in the combo box (case-insensitive)
            index = self.combo_bg_color.findText(cli_args.bg_colour, 
                                                Qt.MatchFlag.MatchFixedString)
            if index >= 0:
                self.combo_bg_color.setCurrentIndex(index)
            else:
                print(f"Warning: Background color '{cli_args.bg_colour}' not found in options.")

        self.setup_keybindings()

        # Debounce the sliders for heavy computation options

        self.shadow_timer = QTimer()
        self.shadow_timer.setSingleShot(True)
        self.shadow_timer.setInterval(50) # Wait 50ms after last movement
        self.shadow_timer.timeout.connect(self.update_output_preview)

        self.trimap_timer = QTimer()
        self.trimap_timer.setSingleShot(True)
        self.trimap_timer.setInterval(100) 
        self.trimap_timer.timeout.connect(self.update_trimap_preview)

        # Let the UI build before loading image, so the correct display zoom is shown
        # 50ms seems long enough on my PC
        if self.image_paths:
            QTimer.singleShot(50, lambda: self.load_image(self.image_paths[0]))
        else:
            self.load_blank_image()
        
        # Delay until UI created
        QTimer.singleShot(10, self.update_cached_model_icons)
        # Delay model pre-loading
        QTimer.singleShot(100, self.preload_startup_models)

        if self.image_paths:
            self.update_thumbnail_strip()
            QTimer.singleShot(50, lambda: self.load_image(self.image_paths[0]))
            
            if cli_args.load_mask:
                QTimer.singleShot(60, lambda: self.load_associated_mask(self.image_paths[0]))
        
        saved_theme = self.settings.value("theme", "light")
        self.set_theme(saved_theme)


        try: pyi_splash.close()
        except: pass



    def init_ui(self):
        self.setAcceptDrops(True)
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        # remove gap above status bar
        layout.setContentsMargins(6, 6, 6, 0) 
        layout.setSpacing(6)

        # --- Sidebar ---
        sidebar_container = QFrame()
        sidebar_container.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        scroll_widget = QWidget()
        controls_layout = QVBoxLayout(scroll_widget) # Main layout for controls
        scroll_widget.setFixedWidth(sidebar_container.width()-8) # Constrain content width. -8 so scrollbar doesnt overlap buttons
        scroll_area.setWidget(scroll_widget)
        sidebar_layout.addWidget(scroll_area)

        # --- Hardware Acceleration Dropdown ---
        self.available_eps = ModelManager.get_available_ep_options()
        
        self.hw_options_frame = CollapsibleFrame("Hardware Acceleration Options", 
                                                 tooltip="Configure which hardware (CPU/GPU) is used for different model types.\n" 
                                                         "First run on GPU can take take while models compile\n"
                                                         "Optimised compiled TensorRT and OpenVINO models are cached to the HDD.\n"
                                                         "Caching models in RAM can speed up subsequent runs."
                                                         )
        hw_layout = self.hw_options_frame.layout_for_content()
        
        # SAM EP Combo
        labelS = QLabel("<b>Interactive SAM Model Provider:</b>")
        labelS.setContentsMargins(3, 0, 0, 0)   
        hw_layout.addWidget(labelS)
        self.combo_sam_model_EP = QComboBox()
        for label, provider_str, opts, short_code in self.available_eps:
            self.combo_sam_model_EP.addItem(label, (provider_str, opts, short_code))

        last_sam = self.settings.value("sam_exec_short_code", "cpu")
        idx = 0 # Default to first item
        for i in range(self.combo_sam_model_EP.count()):
            if self.combo_sam_model_EP.itemData(i)[2] == last_sam:
                idx = i
                break
        self.combo_sam_model_EP.setCurrentIndex(idx)
        self.combo_sam_model_EP.currentIndexChanged.connect(self.on_sam_EP_changed)
        hw_layout.addWidget(self.combo_sam_model_EP)


        #  SAM Caching 
        self.sam_cache_group = QButtonGroup(self)
        self.rb_sam_cache_last = QRadioButton("Keep Last Used In Memory")
        self.rb_sam_cache_all = QRadioButton("Keep All In Memory")
        self.sam_cache_group.addButton(self.rb_sam_cache_last, 1)
        self.sam_cache_group.addButton(self.rb_sam_cache_all, 2)
        self.rb_sam_cache_last.setToolTip("Recommended for good balance between efficiency and memory usage.")
        self.rb_sam_cache_all.setToolTip("Keeps every used SAM model loaded in memory for the session.\nCan cause issues on low VRAM GPUs.")
        hw_layout.addWidget(self.rb_sam_cache_last)
        hw_layout.addWidget(self.rb_sam_cache_all)
        self.sam_cache_group.buttonToggled.connect(self.on_sam_cache_changed)

        last_sam_cache_mode = self.settings.value("sam_ram_cache_mode", 1, type=int)
        self.sam_cache_group.blockSignals(True)
        self.sam_cache_group.button(last_sam_cache_mode).setChecked(True)
        self.sam_cache_group.blockSignals(False)
        self.model_manager.sam_cache_mode = last_sam_cache_mode


        # Automatic Models
        labelW = QLabel("<b>Automatic Model Provider:</b>")
        labelW.setContentsMargins(3, 0, 0, 0)   # push right by 10px
        hw_layout.addWidget(labelW)

        self.combo_auto_model_EP = QComboBox()
        self.combo_auto_model_EP.setToolTip("Select hardware acceleration for automatic background detection models.")
        
        for label, provider_str, opts, short_code in self.available_eps:
            self.combo_auto_model_EP.addItem(label, (provider_str, opts, short_code))
        
        last_exec = self.settings.value("exec_short_code", "cpu")
        idx = 0 # Default to first item
        for i in range(self.combo_auto_model_EP.count()):
            if self.combo_auto_model_EP.itemData(i)[2] == last_exec:
                idx = i
                break
        self.combo_auto_model_EP.setCurrentIndex(idx)
        self.combo_auto_model_EP.currentIndexChanged.connect(self.on_auto_EP_changed)
        hw_layout.addWidget(self.combo_auto_model_EP)
        
        # Auto Model Caching
        self.auto_cache_group = QButtonGroup(self)
        self.rb_auto_cache_none = QRadioButton("Unload after use")
        self.rb_auto_cache_last = QRadioButton("Keep Last Used in Memory")
        self.rb_auto_cache_all = QRadioButton("Keep All In Memory")
        self.auto_cache_group.addButton(self.rb_auto_cache_none, 0)
        self.auto_cache_group.addButton(self.rb_auto_cache_last, 1)
        self.auto_cache_group.addButton(self.rb_auto_cache_all, 2)
        self.rb_auto_cache_none.setToolTip("Only select if memory constrained.")
        self.rb_auto_cache_last.setToolTip("Recommended for good balance between efficiency and memory usage.")
        self.rb_auto_cache_all.setToolTip("Keeps every used automatic model loaded in memory for the session.\nCan cause issues on low VRAM GPUs.")
        hw_layout.addWidget(self.rb_auto_cache_none)
        hw_layout.addWidget(self.rb_auto_cache_last)
        hw_layout.addWidget(self.rb_auto_cache_all)
        self.auto_cache_group.buttonToggled.connect(self.on_auto_cache_changed)

        # Add all to layout
        controls_layout.addWidget(self.hw_options_frame)
        last_auto_cache_mode = self.settings.value("auto_ram_cache_mode", 1, type=int)
        self.auto_cache_group.blockSignals(True)
        self.auto_cache_group.button(last_auto_cache_mode).setChecked(True)
        self.auto_cache_group.blockSignals(False)

        self.model_manager.auto_cache_mode = last_auto_cache_mode

        self.trt_cache_option_visibility() # Initial check for TensorRT

        # End Hardware Acceleration




        nav = QHBoxLayout()
        btn_open_file = QPushButton("Open Image(s)"); btn_open_file.clicked.connect(self.load_image_dialog)
        btn_open_clipboard = QPushButton("Clipboard"); btn_open_clipboard.clicked.connect(self.load_clipboard)
        nav.addWidget(btn_open_file); nav.addWidget(btn_open_clipboard)
        controls_layout.addLayout(nav)


        btn_edit_image = QPushButton("Edit Image"); btn_edit_image.clicked.connect(self.open_image_editor)
        btn_load_mask = QPushButton("Load Mask"); btn_load_mask.clicked.connect(self.load_mask_dialog)

        h_edit_load_buttons = QHBoxLayout()
        h_edit_load_buttons.addWidget(btn_edit_image)
        h_edit_load_buttons.addWidget(btn_load_mask)
        controls_layout.addLayout(h_edit_load_buttons)

        # End File loading buttons

        

        # Mask generation model selection

        h_models_header = QHBoxLayout()
        lbl_models = QLabel("<b> Mask Generation:</b>")
        lbl_models.setContentsMargins(3, 0, 0, 0)
        h_models_header.addWidget(lbl_models)
        h_models_header.addStretch()

        self.btn_download = QPushButton("Download 📥")
        self.btn_download.setToolTip("Download Models...")
        self.btn_download.setFixedSize(120, 32)
        self.btn_download.clicked.connect(self.open_settings)
        h_models_header.addWidget(self.btn_download)
        controls_layout.addLayout(h_models_header)
        
        lbl_sam = QLabel(" Interactive (SAM):")
        lbl_sam.setToolTip("<b>Segment Anything Models</b><br>"
                           "These require you to interact with the image.<br>"
                           "<i>Usage: Left-click to add points, right-click to add negative (avoid) points, or drag to draw boxes around the subject.</i><br><br>"
                           "Disc drive icons show models that have saved optimised versions cached.")
        controls_layout.addWidget(lbl_sam)

        self.combo_sam = QComboBox()
        self.combo_sam.setToolTip(lbl_sam.toolTip())
        self.populate_sam_models()
        controls_layout.addWidget(self.combo_sam)
        lbl_whole = QLabel(" Automatic (Whole Image):")
        lbl_whole.setToolTip("<b>Automatic Models</b><br>"
                             "These run automatically on the entire image.<br>"
                             "<i>Usage: Select a model and click 'Run Automatic'. No points needed.</i><br><br>"
                             "Disc drive icons show models that have saved optimised versions cached.")
        controls_layout.addWidget(lbl_whole)

        # Whole Image Combo
        self.combo_whole = QComboBox()
        self.combo_whole.setToolTip(lbl_whole.toolTip()) # Reuse the tooltip
        self.populate_whole_models()
        controls_layout.addWidget(self.combo_whole)
        
        # Run Model Button and layout adjustment
        h_whole_model = QHBoxLayout()
        self.btn_whole = QPushButton("Run Model"); self.btn_whole.clicked.connect(lambda: self.run_automatic_model())
        h_whole_model.addWidget(self.combo_whole)
        h_whole_model.addWidget(self.btn_whole)
        controls_layout.addLayout(h_whole_model)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(divider)

        h_clr_pnt = QHBoxLayout()
        btn_clr = QPushButton("Clear Points/Masks"); btn_clr.clicked.connect(self.clear_overlay)
        self.chk_paint = QCheckBox("Paintbrush (P)"); self.chk_paint.toggled.connect(self.toggle_paint_mode)
        h_clr_pnt.addWidget(btn_clr); h_clr_pnt.addWidget(self.chk_paint)
        controls_layout.addLayout(h_clr_pnt)


        


        lbl_modifiers = QLabel("<b> Mask Modifiers:</b>")
        lbl_modifiers.setContentsMargins(3, 0, 0, 0)
        controls_layout.addWidget(lbl_modifiers)

        self.chk_post = QCheckBox("Binary Mask (no partial transparency)")
        controls_layout.addWidget(self.chk_post)

        self.chk_soften = QCheckBox("Soften Mask/Paintbrush Edges")
        soften_checked = self.settings.value("soften_mask", False, type=bool)
        self.chk_soften.setChecked(soften_checked)
        self.chk_soften.toggled.connect(lambda checked: self.settings.setValue("soften_mask", checked))
        controls_layout.addWidget(self.chk_soften)
        
        self.chk_alpha_matting = QCheckBox("Alpha Matting (Slow, Experimental)")
        self.chk_alpha_matting.setToolTip("Uses a matting algorithm to estimate the transparency of mask edges.\n"
                                          "This can improve the quality of detailed edges such as hair, especially when using binary mask models like SAM.\n"
                                          "This requires a trimap, either estimated from a SAM or automatic models, or manually drawn.\n"
                                          "This is computationally expensive and is only applied when 'Add' or 'Subtract' is clicked. Undo if the effect is unsatisfactory")
        controls_layout.addWidget(self.chk_alpha_matting)

        self.chk_alpha_matting.toggled.connect(self.handle_alpha_matting_toggle)
        controls_layout.addWidget(self.chk_alpha_matting)

        self.chk_alpha_matting.toggled.connect(self.handle_alpha_matting_toggle)
        controls_layout.addWidget(self.chk_alpha_matting)

        # --- Alpha Matting Options Frame ---
        self.alpha_matting_frame = QFrame()
        am_layout = QVBoxLayout(self.alpha_matting_frame)
        am_layout.setContentsMargins(15, 5, 0, 5) # Indent options slightly

        matting_label = QLabel("<b>Matting Algorithm:</b>")
        matting_tt = "Additional models can be downloaded using the model manager.\nThe default included PyMatting algo can be very slow on large images.\nViTMatte (model downloader) can be much faster and far more accurate"
        matting_label.setToolTip(matting_tt)
        am_layout.addWidget(matting_label)

        self.combo_matting_algorithm = QComboBox()
        self.combo_matting_algorithm.setToolTip(matting_tt)
        self.populate_matting_models()
        am_layout.addWidget(self.combo_matting_algorithm)

        # --- Trimap Source Radio Buttons ---
        am_layout.addWidget(QLabel("<b>Trimap Source:</b>"))
        self.trimap_mode_group = QButtonGroup(self)
        self.rb_trimap_auto = QRadioButton("Automatic (from model mask + sliders)")
        self.rb_trimap_custom = QRadioButton("Custom (user-drawn)")
        
        self.trimap_mode_group.addButton(self.rb_trimap_auto)
        self.trimap_mode_group.addButton(self.rb_trimap_custom)
        
        self.rb_trimap_auto.setChecked(True)

        am_layout.addWidget(self.rb_trimap_auto)
        am_layout.addWidget(self.rb_trimap_custom)
        
        self.trimap_mode_group.buttonToggled.connect(self.on_trimap_mode_changed)
        
        # --- Group sliders for easy show/hide ---
        self.auto_trimap_sliders_widget = QWidget()
        sliders_layout = QVBoxLayout(self.auto_trimap_sliders_widget)
        sliders_layout.setContentsMargins(0, 5, 0, 0)
        
        def make_am_slider_row(lbl_text, min_v, max_v, def_v):
            h_layout = QHBoxLayout()
            label = QLabel(f"{lbl_text}: {def_v}")
            label.setMinimumWidth(120)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(def_v)
            slider.valueChanged.connect(lambda val, l=label, txt=lbl_text: (l.setText(f"{txt}: {val}"), self.update_trimap_preview_throttled()))
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            return label, slider, h_layout

        self.lbl_fg_erode, self.sl_fg_erode, fg_layout = make_am_slider_row("FG Shrink", 0, 100, 15)
        self.sl_fg_erode.setToolTip("Shrinks the solid foreground area to create the 'unknown' region.")
        sliders_layout.addLayout(fg_layout)

        self.lbl_bg_erode, self.sl_bg_erode, bg_layout = make_am_slider_row("BG Shrink", 0, 100, 15)
        self.sl_bg_erode.setToolTip("Shrinks the solid background area to create the 'unknown' region.")
        sliders_layout.addLayout(bg_layout)

        am_layout.addWidget(self.auto_trimap_sliders_widget)
        
        self.btn_edit_trimap = QPushButton("Open Trimap Editor...")
        self.btn_edit_trimap.clicked.connect(self.open_trimap_editor)
        self.btn_edit_trimap.setVisible(False)
        am_layout.addWidget(self.btn_edit_trimap)
        
        self.chk_show_trimap = QCheckBox("Show Trimap on Input")
        self.chk_show_trimap.setToolTip("Displays the generated trimap on the input view.\n"
                                        "White = Foreground, Blue = Unknown (semi transparent, e.g. hair edges), Black = Background")
        self.chk_show_trimap.toggled.connect(self.toggle_trimap_display)
        am_layout.addWidget(self.chk_show_trimap)


        controls_layout.addWidget(self.alpha_matting_frame)
        self.alpha_matting_frame.hide()

        # end alpha matting



        lbl_actions = QLabel("<b> Output Image Actions:</b>")
        lbl_actions.setContentsMargins(3, 0, 0, 0)
        controls_layout.addWidget(lbl_actions)
        h_act = QHBoxLayout()
        btn_add = QPushButton("Add Mask (A)"); btn_add.clicked.connect(self.add_mask)
        btn_add.setToolTip("Add the current model output mask to the composite output image." 
                           "Mask refinement steps e.g. alpha matting are added at this step")
        btn_sub = QPushButton("Sub Mask (S)"); btn_sub.clicked.connect(self.subtract_mask)
        btn_sub.setToolTip("Subtract the current model output mask to the composite output image." 
                           "Mask refinement steps e.g. alpha matting are added at this step")
        h_act.addWidget(btn_add); h_act.addWidget(btn_sub)
        controls_layout.addLayout(h_act)
        
        h_ut = QHBoxLayout()
        btn_undo = QPushButton("Undo ↶")
        btn_undo.clicked.connect(self.undo)
        btn_undo.setToolTip("Undo")

        btn_redo = QPushButton("Redo ↷")
        btn_redo.clicked.connect(self.redo)
        btn_redo.setToolTip("Redo")

        h_ut.addWidget(btn_undo); h_ut.addWidget(btn_redo)
        controls_layout.addLayout(h_ut)



        lbl_canvas = QLabel("<b> Canvas:</b>")
        lbl_canvas.setContentsMargins(3, 0, 0, 0)
        controls_layout.addWidget(lbl_canvas)



        
        h_rs = QHBoxLayout()
        btn_rst = QPushButton("Reset Img"); btn_rst.clicked.connect(self.reset_working_image)
        btn_all = QPushButton("Reset All"); btn_all.clicked.connect(self.reset_all)
        h_rs.addWidget(btn_rst); h_rs.addWidget(btn_all)
        controls_layout.addLayout(h_rs)
        
        h_vs = QHBoxLayout()
        btn_cp = QPushButton("Copy In->Out"); btn_cp.clicked.connect(self.copy_input_to_output)
        btn_c_vis = QPushButton("Clear Viewport"); btn_c_vis.clicked.connect(self.clear_visible_area)
        h_vs.addWidget(btn_cp); h_vs.addWidget(btn_c_vis) 
        controls_layout.addLayout(h_vs)

        lbl_options = QLabel("<b>Output Styling:</b>")
        lbl_options.setContentsMargins(3, 0, 0, 0)
        controls_layout.addWidget(lbl_options)

        bg_layout = QHBoxLayout()
        bg_label = QLabel("Background:")
        bg_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        bg_layout.addWidget(bg_label)
        self.combo_bg_color = QComboBox()
        self.combo_bg_color.addItems(["Transparent", "White", "Black", "Red", "Blue", 
                  "Orange", "Yellow", "Green", "Grey", 
                  "Lightgrey", "Brown", "Blurred (Slow)"])
        self.combo_bg_color.currentTextChanged.connect(self.handle_bg_change)
        bg_layout.addWidget(self.combo_bg_color)
        controls_layout.addLayout(bg_layout)
        

        self.chk_estimate_foreground = QCheckBox("Mask Edge Colour Correction (Slow)")
        self.chk_estimate_foreground.setToolTip("Recalculates edge colors to remove halos or fringes from the original background.\n"
                                                "Recommended for soft edges such as hair")
        controls_layout.addWidget(self.chk_estimate_foreground)
        self.chk_estimate_foreground.toggled.connect(self.update_output_preview)


        self.chk_shadow = QCheckBox("Drop Shadow")
        self.chk_shadow.toggled.connect(self.toggle_shadow_options)
        controls_layout.addWidget(self.chk_shadow)

        self.shadow_frame = QFrame()
        sf_layout = QVBoxLayout(self.shadow_frame)
        sf_layout.setContentsMargins(0,0,0,0)
        
        def make_slider_row(lbl_text, min_v, max_v, def_v):
            h_layout = QHBoxLayout()
            
            l = QLabel(f"{lbl_text}: {def_v}")
            l.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
            l.setMinimumWidth(80)
            
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(min_v, max_v)
            s.setValue(def_v)
            s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            
            s.valueChanged.connect(lambda v: (l.setText(f"{lbl_text}: {v}"), self.shadow_timer.start()))
            
            h_layout.addWidget(l)
            h_layout.addWidget(s)
            
            return l, s, h_layout

        self.lbl_s_op, self.sl_s_op, h_op_layout = make_slider_row("Opacity", 0, 255, 128)
        self.lbl_s_x, self.sl_s_x, h_x_layout = make_slider_row("X Offset", -100, 100, 30)
        self.lbl_s_y, self.sl_s_y, h_y_layout = make_slider_row("Y Offset", -100, 100, 30)
        self.lbl_s_r, self.sl_s_r, h_r_layout = make_slider_row("Blur Rad", 1, 50, 10)
        
        sf_layout.addLayout(h_op_layout)
        sf_layout.addLayout(h_x_layout)
        sf_layout.addLayout(h_y_layout)
        sf_layout.addLayout(h_r_layout)
            
        controls_layout.addWidget(self.shadow_frame)
        self.shadow_frame.hide()

        controls_layout.addStretch()
        btn_save = QPushButton("Save As..."); btn_save.clicked.connect(self.save_image)
        controls_layout.addWidget(btn_save)
        btn_qsave = QPushButton("Quick Save JPG"); btn_qsave.clicked.connect(self.quick_save_jpeg)
        controls_layout.addWidget(btn_qsave)

        bottom_buttons_layout = QHBoxLayout()

        btn_help = QPushButton("Help / About"); btn_help.clicked.connect(self.show_help)
        bottom_buttons_layout.addWidget(btn_help)

        self.btn_settings = QPushButton("⚙️")
        self.btn_settings.setToolTip("Settings / Model Manager")
        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_settings.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        bottom_buttons_layout.addWidget(self.btn_settings)
        controls_layout.addLayout(bottom_buttons_layout)

        # end Sidebar


        self.in_out_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.in_out_splitter.setChildrenCollapsible(False)
        
        is_light_mode = QApplication.styleHints().colorScheme() != Qt.ColorScheme.Dark
        hatch_color = QColor(220, 220, 220) if is_light_mode else QColor(60, 60, 60)

        w_in = QWidget(); l_in = QVBoxLayout(w_in)
        w_in.setMinimumWidth(150)

        # Add invisible widget to match the splitter toggle widget to keep layout spacing correct
        h_in_header = QHBoxLayout()
        h_in_header.addWidget(QLabel("Interactive Input. Models are run on the current viewport. Zoom for greater detail"))
        h_in_header.addStretch()
        header_spacer = QWidget()
        header_spacer.setFixedSize(24, 24)
        h_in_header.addWidget(header_spacer)
        
        l_in.addLayout(h_in_header)
        self.scene_input = QGraphicsScene()
        self.view_input = SynchronisedGraphicsView(self.scene_input, name="Input View") 
        self.view_input.set_controller(self)
        self.view_input.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))
        self.view_input.setAcceptDrops(True)


        self.input_pixmap_item = QGraphicsPixmapItem(); self.scene_input.addItem(self.input_pixmap_item)
        self.overlay_pixmap_item = QGraphicsPixmapItem(); self.overlay_pixmap_item.setOpacity(0.5); self.scene_input.addItem(self.overlay_pixmap_item)
        
        self.trimap_overlay_item = QGraphicsPixmapItem(); self.trimap_overlay_item.setOpacity(0.7); self.scene_input.addItem(self.trimap_overlay_item)

        l_in.addWidget(self.view_input)
        self.in_out_splitter.addWidget(w_in)
        
        w_out = QWidget(); l_out = QVBoxLayout(w_out)
        w_out.setMinimumWidth(150)

        # Output header with split orientation change button
        h_out_header = QHBoxLayout()
        h_out_header.addWidget(QLabel("Output Composite"))
        h_out_header.addStretch()
        self.chk_show_mask = QCheckBox("Show Mask"); self.chk_show_mask.toggled.connect(self.update_output_preview)
        h_out_header.addWidget(self.chk_show_mask)
        self.toggle_split_button = QPushButton()
        self.toggle_split_button.setFlat(True)
        self.toggle_split_button.setFixedSize(24, 24)
        self.toggle_split_button.setToolTip("Toggle between vertical and horizontal split view")
        self.toggle_split_button.clicked.connect(self.toggle_splitter_orientation)
        h_out_header.addWidget(self.toggle_split_button)
        l_out.addLayout(h_out_header)

        self.scene_output = QGraphicsScene()
        self.view_output = SynchronisedGraphicsView(self.scene_output, name="Output View")
        self.view_output.set_controller(self) 
        self.view_output.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))
        self.view_output.setAcceptDrops(True)
        
        self.output_pixmap_item = QGraphicsPixmapItem()
        self.scene_output.addItem(self.output_pixmap_item)

        l_out.addWidget(self.view_output)
        self.in_out_splitter.addWidget(w_out)

        self.view_input.set_sibling(self.view_output)
        self.view_output.set_sibling(self.view_input)

        # Holds the views (top) and the thumbnails (bottom)
        views_thumbs_widget = QWidget()
        views_thumbs_layout = QVBoxLayout(views_thumbs_widget)
        views_thumbs_layout.setContentsMargins(0, 0, 0, 0)
        views_thumbs_layout.setSpacing(2) # Small gap between views and thumbnails

        # Initialise the thumbnail strip
        self.thumbnail_strip = ThumbnailList()
        self.thumbnail_strip.itemClicked.connect(self.on_thumbnail_clicked)

        # Add the splitter to the top, thumbnails to the bottom
        views_thumbs_layout.addWidget(self.in_out_splitter, 1) # '1' ensures views take most space
        views_thumbs_layout.addWidget(self.thumbnail_strip)





        # Assemble sidebar + views
        layout.addWidget(sidebar_container)
        layout.addWidget(views_thumbs_widget, 1) # Add the container instead of the splitter

        # status bar 
        self.status_label = QLabel("Idle")
        self.status_label.setFixedWidth(600)
        self.zoom_label = QLabel("Zoom: 100%")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Sets "Indeterminate" (bouncing) mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.hide()

        self.statusBar().addWidget(self.status_label)
        self.statusBar().addPermanentWidget(self.progress_bar) # Add to right side
        self.statusBar().addPermanentWidget(self.zoom_label)
        
        self.toggle_splitter_orientation(initial_setup=True)

    def is_mask_modified(self):
        """Checks if the working mask has data or history."""
        if len(self.undo_history) > 1:
            return True
        if self.working_mask:
            # Check for non-zero pixels
            extrema = self.working_mask.getextrema()
            if extrema and extrema[1] > 0:
                return True
        return False

    def check_proceed_if_modified(self):
        """Prompts user if mask is modified. Returns True to proceed."""
        if self.is_mask_modified():
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "The current mask has been modified. Switching images will clear the mask. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    def on_thumbnail_clicked(self, item):
        new_index = self.thumbnail_strip.row(item)
        if new_index == self.current_image_index:
            return

        if self.check_proceed_if_modified():
            self.current_image_index = new_index
            path = item.data(Qt.ItemDataRole.UserRole)
            if path == "Clipboard":
                self.load_clipboard()
            else:
                self.load_image(path)

    def update_thumbnail_strip(self):
        """Synchronizes the list widget with self.image_paths."""
        self.thumbnail_strip.clear()
        for path in self.image_paths:
            self.thumbnail_strip.add_image_thumbnail(path)
            # Could possibly thread this, but individual image loading is fast,
            # so just show them one by one as they load to prevent app hanging
            QApplication.processEvents()
        
        if self.current_image_index < self.thumbnail_strip.count():
            self.thumbnail_strip.setCurrentRow(self.current_image_index)

    def set_theme(self, mode):
        """Applies light or dark mode and saves the preference.
        Unfortunately QApplication.StyleHints.setStylesheet(Qt.Color.Dark)
        doesn't change the theme after widgets are drawn (outside of if name==main)
        So manual theming is required
        """
        app = QApplication.instance()
        if not app: return

        if mode == 'dark':
            app.setStyle("Fusion")
            dark_palette = QPalette()
            dark_color = QColor(45, 45, 45)
            disabled_color = QColor(127, 127, 127)
            dark_palette.setColor(QPalette.ColorRole.Window, dark_color)
            dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
            dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.Button, dark_color)
            dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
            dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
            dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
            dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
            dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
            dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
            dark_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, disabled_color)
            app.setPalette(dark_palette)

            hatch_color = QColor(60, 60, 60)
            self.view_input.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))
            self.view_output.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))

            self.settings.setValue("theme", "dark")
        else: # light mode
            # manually set because windows sets default as black when system is black. opposite to linux....
            app.setStyle("Fusion")
            light_palette = QPalette()
            light_color = QColor(240, 240, 240)            
            light_palette.setColor(QPalette.ColorRole.Window, light_color)
            light_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
            light_palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
            light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
            light_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
            light_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
            light_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
            light_palette.setColor(QPalette.ColorRole.Button, light_color)
            light_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
            light_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
            light_palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
            light_palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
            light_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
            disabled_text = QColor(120, 120, 120)
            light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
            light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
            light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
            light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(150, 150, 150))
            light_palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)

            app.setPalette(light_palette)
            
            hatch_color = QColor(220, 220, 220)
            self.view_input.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))
            self.view_output.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))
            
            self.settings.setValue("theme", "light")

        # update widgets that require overrides
        self.hw_options_frame.collapsible_set_light_dark()
        self.toggle_splitter_orientation(initial_setup=True)
        self.thumbnail_strip.update_style(mode == 'dark')


    def toggle_splitter_orientation(self, initial_setup=False):
        current_orientation = self.in_out_splitter.orientation()
        
        if initial_setup:
            target_orientation = current_orientation
        else:
            target_orientation = Qt.Orientation.Vertical if current_orientation == Qt.Orientation.Horizontal else Qt.Orientation.Horizontal
            self.in_out_splitter.setOrientation(target_orientation)

        icon_pixmap_enum = QStyle.StandardPixmap.SP_ToolBarHorizontalExtensionButton if target_orientation == Qt.Orientation.Vertical else QStyle.StandardPixmap.SP_ToolBarVerticalExtensionButton
        
        is_dark_mode = app.palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark_mode:
            icon = self.style().standardIcon(icon_pixmap_enum)
            pixmap = icon.pixmap(24, 24) 

            img = pixmap.toImage()

            # InvertRgb so we don't invert the alpha channel
            img.invertPixels(QImage.InvertMode.InvertRgb)
            
            self.toggle_split_button.setIcon(QIcon(QPixmap.fromImage(img)))
        else:
            self.toggle_split_button.setIcon(self.style().standardIcon(icon_pixmap_enum))

    
    

    
    def _get_cached_icon(self):
        """
        Returns a standard system icon to indicate a model is cached on disk.
        Returns a cached QIcon to avoid fetching it every time.
        """
        if hasattr(self, "_cached_drive_icon"):
            return self._cached_drive_icon

        # Fetch the standard "Hard Drive" icon from the current application style
        self._cached_drive_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
        
        return self._cached_drive_icon

    def update_cached_model_icons(self):
        """Update drive icons based on disk cache."""
        # Check Whole Image Models
        current_data = self.combo_auto_model_EP.currentData() # (ProviderStr, Opts, ShortCode)
        if current_data:
            short_code = current_data[2]
            drive_icon = self._get_cached_icon() 
            
            for i in range(self.combo_whole.count()):
                m_name = self.combo_whole.itemText(i)
                if self.model_manager.check_is_cached(m_name, short_code):
                    self.combo_whole.setItemIcon(i, drive_icon)
                else:
                    self.combo_whole.setItemIcon(i, QIcon()) # Set blank icon if not cached

        # Check SAM Models (Same logic)
        current_sam_data = self.combo_sam_model_EP.currentData()
        if current_sam_data:
            sam_code = current_sam_data[2]
            drive_icon = self._get_cached_icon()
            
            for i in range(self.combo_sam.count()):
                m_name = self.combo_sam.itemText(i)
                if self.model_manager.check_is_cached(m_name, sam_code):
                    self.combo_sam.setItemIcon(i, drive_icon)
                else:
                    self.combo_sam.setItemIcon(i, QIcon())

    def on_auto_EP_changed(self, index):
        """
        Handle change in automatic image execution provider.
        """
        # Retrieve from combobox userdata (ProviderStr, OptionsDict, ShortCode)
        data = self.combo_auto_model_EP.itemData(index)
        if not data: return
        prov_str, prov_opts, short_code = data

        self.settings.setValue("exec_short_code", short_code)
        
        self.model_manager.clear_auto_cache()
        
        self.update_cached_model_icons()

        self.status_label.setText(f"Automatic model provider switched to: {short_code.upper()}")
        self.trt_cache_option_visibility()

    def trt_cache_option_visibility(self):
        """
        Disables 'No Caching' if TensorRT is selected for auto models.
        Unsure if this should be also used for OpenVINO, but for now allow unloading
        """
        data = self.combo_auto_model_EP.currentData()
        if not data: return
        prov_str, _, _ = data

        is_trt = (prov_str == "TensorrtExecutionProvider")
        self.rb_auto_cache_none.setEnabled(not is_trt)
        self.rb_auto_cache_none.setVisible(not is_trt)

        # If TRT is selected and "No Cache" was active, switch to "Keep Last"
        if is_trt and self.rb_auto_cache_none.isChecked():
            self.rb_auto_cache_last.setChecked(True)


    def on_auto_cache_changed(self, button, checked):
        if checked:
            # Clear loaded models regardless of option chosen
            # to allow easy unloading and consistent behaviour
            self.model_manager.clear_auto_cache()
            self.status_label.setText("Automatic model cache cleared.")

            cache_mode = self.auto_cache_group.id(button)

            if cache_mode == 2: # Keep all
                agreed = self.settings.value("agreed_high_vram_warning", False, type=bool)
                if not agreed:
                    ret = QMessageBox.warning(self, "High VRAM Warning",
                        "Keeping multiple used models in memory with GPU acceleration can fill VRAM "
                        "and cause the application to crash.\n\n"
                        "Are you sure you want to enable this?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if ret == QMessageBox.StandardButton.Yes:
                        self.settings.setValue("agreed_high_vram_warning", True)
                    else:
                        # Revert to "Keep Last"
                        self.rb_auto_cache_last.setChecked(True)
                        return

            self.settings.setValue("auto_ram_cache_mode", cache_mode)
            self.model_manager.auto_cache_mode = cache_mode

    def on_sam_cache_changed(self, button, checked):
        if checked:
            # Clear loaded models regardless of option chosen
            # to allow easy unloading and consistent behaviour
            self.model_manager.clear_sam_cache()
            self.status_label.setText("SAM model cache cleared.")

            cache_mode = self.sam_cache_group.id(button)

            if cache_mode == 2: # Keep all
                agreed = self.settings.value("agreed_high_vram_warning", False, type=bool)
                if not agreed:
                    ret = QMessageBox.warning(self, "High VRAM Warning",
                        "Keeping multiple used models in memory with GPU acceleration can fill VRAM "
                        "and cause the application to crash.\n\n"
                        "Are you sure you want to enable this?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if ret == QMessageBox.StandardButton.Yes:
                        self.settings.setValue("agreed_high_vram_warning", True)
                    else:
                        # Revert to "Keep Last"
                        self.rb_sam_cache_last.setChecked(True)
                        return

            self.settings.setValue("sam_ram_cache_mode", self.sam_cache_group.id(button))
            self.model_manager.sam_cache_mode = cache_mode

    def on_sam_EP_changed(self, index):
        """
        Handle change in SAM execution provider.
        """
        # Retrieve from combobox userdata (ProviderStr, OptionsDict, ShortCode)
        data = self.combo_sam_model_EP.itemData(index)
        if not data: return
        prov_str, prov_opts, short_code = data

        self.settings.setValue("sam_exec_short_code", short_code)
        
        self.model_manager.clear_sam_cache()
        self.update_cached_model_icons()

        self.status_label.setText(f"SAM provider switched to: {short_code.upper()}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            supported_exts = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff']
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if os.path.splitext(path)[1].lower() in supported_exts:
                        event.acceptProposedAction()
                        return
        super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            supported_exts = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff']
            fnames = [url.toLocalFile() for url in urls if url.isLocalFile() and os.path.splitext(url.toLocalFile())[1].lower() in supported_exts]
            
            if fnames:
                self.image_paths = sorted(fnames)
                self.current_image_index = 0
                self.update_thumbnail_strip()
                self.load_image(self.image_paths[0])
        super().dropEvent(event)

    def update_window_title(self):
        base_title = "Interactive Image Background Remover"
        
        if not self.image_paths:
            self.setWindowTitle(base_title)
            return

        total_count = len(self.image_paths)
        
        if total_count == 1 and self.image_paths[0] == "Clipboard":
            title = f"{base_title} [Clipboard]"
        elif total_count > 1:
            current = self.current_image_index + 1
            # Show the count and the current file name
            file_name = os.path.basename(self.image_paths[self.current_image_index])
            title = f"{base_title} [{current}/{total_count}] - {file_name}"
        else: # Single image from file/arg
            file_name = os.path.basename(self.image_paths[0])
            title = f"{base_title} - {file_name}"

        self.setWindowTitle(title)

    def set_loading(self, is_loading, message=None):
        if is_loading:
            self.progress_bar.show()
            self.status_label.setText(message if message else "Processing...")
            self.status_label.setStyleSheet("color: red;")
            # Prevent multiple threads being created by user clicking stuff
            self.centralWidget().layout().itemAt(0).widget().setEnabled(False)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents() 
        else:
            self.progress_bar.hide()
            self.status_label.setText(message if message else "Idle")
            self.status_label.setStyleSheet("")
            self.centralWidget().layout().itemAt(0).widget().setEnabled(True)
            QApplication.restoreOverrideCursor()

    def showEvent(self, event):
        super().showEvent(event)
        w = self.in_out_splitter.width()
        self.in_out_splitter.setSizes([w//2, w//2])

    def open_settings(self):
        dlg = settings_download_manager.SettingsDialog(
            model_root_dir=self.model_root_dir, 
            main_app_instance=self, 
            parent=self
        )
        dlg.exec()

    def update_model_root_dir(self, new_dir):
        """Updates the path in the main application and synchronises the model manager."""
        self.model_root_dir = new_dir
        self.model_manager.model_root_dir = new_dir
        
        self.populate_sam_models()
        self.populate_whole_models()
        self.populate_matting_models()
        self.update_cached_model_icons()

    def populate_sam_models(self):
        sam_models = ["mobile_sam", "sam_vit_b", "sam_vit_h", "sam_vit_l", "sam2"]
        matches = []
        if os.path.exists(self.model_root_dir):
            for filename in os.listdir(self.model_root_dir):
                for partial in sam_models:
                    if partial in filename and ".onnx" in filename:
                        matches.append(filename.replace(".encoder.onnx","").replace(".decoder.onnx",""))
        self.combo_sam.clear()
        if matches:
            self.combo_sam.addItems(sorted(list(set(matches))))
            # Mobile SAM has near instant results even on CPU, so we set as the default if available
            idx = self.combo_sam.findText("mobile_sam", Qt.MatchFlag.MatchContains)
            if idx >= 0: self.combo_sam.setCurrentIndex(idx)
        else: self.combo_sam.addItem("No Models Found")

    def populate_whole_models(self):
        # should change these to read from the download manager, but this is easiest for me testing variations of new models atm
        whole_models = ["rmbg", "isnet", "u2net", "BiRefNet", "ben2", "mvanet", "modnet_portrait"]
        matches = []
        if os.path.exists(self.model_root_dir):
            for filename in os.listdir(self.model_root_dir):
                for partial in whole_models:
                    if partial in filename and ".onnx" in filename:
                        matches.append(filename.replace(".onnx",""))
        self.combo_whole.clear()
        if matches: self.combo_whole.addItems(sorted(list(set(matches))))
        else: self.combo_whole.addItem("No Models Found")

    def populate_matting_models(self):
        self.combo_matting_algorithm.clear()

        # Scan for ViTMatte (and in the future, more models)
        matting_models = ["vitmatte"]
        if os.path.exists(self.model_root_dir):
            for filename in os.listdir(self.model_root_dir):
                for partial in matting_models:
                    if partial in filename and filename.endswith(".onnx"):
                        model_name = filename.replace(".onnx", "")
                        self.combo_matting_algorithm.addItem(model_name)

        # Add the older and weaker indexnet model below Vitmatte, if present
        if os.path.exists(os.path.join(self.model_root_dir, "indexnet.onnx")):
            self.combo_matting_algorithm.addItem("indexnet")
        
        # PyMatting is always an option, and often the worst option
        # but i'm importing anyway for estimate_foreground_ml, may as well offer estimate_alpha_cf here
        self.combo_matting_algorithm.addItem("PyMatting (CPU)")

    def setup_keybindings(self):
        QShortcut(QKeySequence("A"), self).activated.connect(self.add_mask)
        QShortcut(QKeySequence("S"), self).activated.connect(self.subtract_mask)
        QShortcut(QKeySequence("C"), self).activated.connect(self.clear_overlay)
        QShortcut(QKeySequence("W"), self).activated.connect(self.reset_working_image)
        QShortcut(QKeySequence("R"), self).activated.connect(self.reset_all)
        QShortcut(QKeySequence("V"), self).activated.connect(self.clear_visible_area)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo)
        QShortcut(QKeySequence("P"), self).activated.connect(self.chk_paint.toggle)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_image)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(self.quick_save_jpeg) # Quick Save JPG
        
        QShortcut(QKeySequence("U"), self).activated.connect(lambda: self.run_automatic_model("u2net"))
        QShortcut(QKeySequence("I"), self).activated.connect(lambda: self.run_automatic_model("isnet-general-use"))
        QShortcut(QKeySequence("O"), self).activated.connect(lambda: self.run_automatic_model("rmbg1_4"))
        QShortcut(QKeySequence("B"), self).activated.connect(lambda: self.run_automatic_model("ben2_base"))

    def handle_bg_change(self, text):
        if "Blur" in text:
            val, ok = QInputDialog.getInt(self, "Blur Radius", "Set Blur Radius:", 
                                         value=getattr(self, 'blur_radius', 30), 
                                         min=1, max=100)
            if ok:
                self.blur_radius = val
        self.update_output_preview()

    def load_image(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            self.original_image = img.convert("RGBA")
            self.image_exif = img.info.get('exif')
            self.init_working_buffers()
            self.update_input_view()
            self.update_output_preview()
            self.status_label.setText(f"Loaded: {os.path.basename(path)} [{self.current_image_index + 1}/{len(self.image_paths)}]")
            self.update_window_title()

            self.thumbnail_strip.blockSignals(True)
            self.thumbnail_strip.setCurrentRow(self.current_image_index)
            self.thumbnail_strip.blockSignals(False)

        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def load_associated_mask(self, image_path):
        mask_path = os.path.splitext(image_path)[0] + "_mask.png"
        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                if mask.size == self.original_image.size:
                    self.add_undo_step()
                    self.working_mask = mask
                    self.update_output_preview()
                    self.status_label.setText(f"Loaded associated mask: {os.path.basename(mask_path)}")
                else:
                     QMessageBox.warning(self, "Mask Size Mismatch", "The associated mask's dimensions do not match the base image.")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Mask", f"Could not load the associated mask:\n{e}")

    def load_blank_image(self):
        self.original_image = Image.new("RGBA", (800, 600), (0,0,0,0))
        self.init_working_buffers()
        self.update_input_view()

    def load_clipboard(self):
        img = ImageGrab.grabclipboard()

        if isinstance(img, Image.Image):
            self.original_image = img.convert("RGBA")
            self.image_paths = ["Clipboard"]
            self.init_working_buffers()
            self.update_input_view()
            self.update_output_preview()
            self.update_thumbnail_strip()
            self.status_label.setText("Loaded from Clipboard [1/1]")
            self.update_window_title()
            return

        QMessageBox.information(self, "Clipboard Empty", 
                                "No image or valid image path found on the clipboard.")

    def load_image_dialog(self):
        file_filter = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.WEBP *.BMP *.TIF *.TIFF);;All Files (*)"
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open", "", file_filter)
        if fnames:
            self.image_paths = fnames
            self.current_image_index = 0
            self.update_thumbnail_strip()
            self.load_image(fnames[0])

    def load_mask_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Mask", "", "Images (*.png)")
        if fname:
            mask = Image.open(fname).convert("L")
            if mask.size == self.original_image.size:
                self.add_undo_step()
                self.working_mask = mask
                self.update_output_preview()
    
    def preload_startup_models(self):
        """
        Loads models selected in the download manager on startup.
        Ideally should be threaded but if user selects small models, its only a second or two to load
        """
        sam_model_id = self.settings.value("startup_sam_model", None)
        auto_model_id = self.settings.value("startup_automatic_general_model", None)

        if sam_model_id:
            # Find the full model name from the ID
            idx = self.combo_sam.findText(sam_model_id, Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.combo_sam.setCurrentIndex(idx)
                
                self.set_loading(True, f"Pre-loading SAM: {self.combo_sam.currentText()}")
                # Call manager to load
                provider_data = self.combo_sam_model_EP.currentData() or ("CPUExecutionProvider", {}, "cpu")
                self.model_manager.init_sam_session(self.combo_sam.currentText(), provider_data)
                self.set_loading(False, f"Pre-loaded SAM: {self.combo_sam.currentText()}")

        if auto_model_id:
            idx = self.combo_whole.findText(auto_model_id, Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.combo_whole.setCurrentIndex(idx)
                self.set_loading(True, f"Pre-loading Automatic Model: {self.combo_whole.currentText()} (Startup)")
                provider_data = self.combo_auto_model_EP.currentData()
                self.model_manager.get_auto_session(self.combo_whole.currentText(), provider_data)
                self.set_loading(False, f"Pre-loaded Automatic Model")

        
    def _sanitise_filename_for_windows(self, path: str) -> str:

        """
        On Windows, strip invalid filename characters from the basename:
        \\/:*?"<>|
        Directory part (e.g. C:\\folder) is left intact.
        """
        if os.name != "nt":
            return path

        directory, basename = os.path.split(path)
        invalid_chars = r'\/:*?"<>|'
        cleaned = ''.join(c for c in basename if c not in invalid_chars)

        if not cleaned:
            cleaned = "output"

        return os.path.join(directory, cleaned)


    def init_working_buffers(self):
        size = self.original_image.size 
        self.working_image = Image.new("RGBA", size, (0,0,0,0))
        self.working_mask = Image.new("L", size, 0)
        self.model_output_mask = Image.new("L", size, 0)
        self.undo_history = [self.working_mask.copy()]
        self.redo_history = []

        # used in render_output_image to cache the colour corrected foreground for quick on/off previewing
        self.working_mask_hash=None
        self.cached_fg_corrected_cutout=None

        self.cached_blurred_bg = None
        self.last_blur_params = None # Stores (mask_hash, blur_radius)
        
        self.last_trimap = None
        if hasattr(self, 'user_trimap'):
            del self.user_trimap

        self.model_manager.clear_sam_cache(clear_loaded_models=False)
        
        # Scratchpad for painting (Invisible QImage)
        # Used to rasterise the vector path when you let go of the mouse.
        self.paint_image = QImage(size[0], size[1], QImage.Format.Format_ARGB32_Premultiplied)
        
        self.clear_overlay()

    def update_input_view(self):
        if self.original_image:
            # 1. Update the Pixmap
            self.input_pixmap_item.setPixmap(pil2pixmap(self.original_image))
            rect = self.input_pixmap_item.boundingRect()
            self.view_input.setSceneRect(rect)

            self.view_input.resetTransform()

            self.view_input.fitInView(self.input_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

            # Set to fill the screen, but could add small padding if preferred
            self.view_input.scale(1, 1)

            if self.view_output:
                self.view_output.setSceneRect(rect)
                self.view_output.setTransform(self.view_input.transform())
            
            self.update_zoom_label()

    def update_zoom_label(self):
        zoom = self.view_input.transform().m11() * 100
        self.zoom_label.setText(f"Zoom: {int(zoom)}%")

    def get_viewport_crop(self):
        vp = self.view_input.viewport().rect()
        sr = self.view_input.mapToScene(vp).boundingRect()
        ir = QRectF(0, 0, self.original_image.width, self.original_image.height)
        cr = sr.intersected(ir)
        x, y, w, h = int(cr.x()), int(cr.y()), int(cr.width()), int(cr.height())
        if w <= 0 or h <= 0: return self.original_image, 0, 0
        return self.original_image.crop((x, y, x+w, y+h)), x, y

    

    def handle_sam_point(self, scene_pos, is_positive):
        
        if self.is_busy():
            return # Interaction blocked during inference
        
        self.coordinates.append([scene_pos.x(), scene_pos.y()])
        self.labels.append(1 if is_positive else 0)
        c = Qt.GlobalColor.green if is_positive else Qt.GlobalColor.red
        
        radius = 5  # Base radius (on screen pixels)
        dot = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
        
        dot.setPos(scene_pos)

        dot.setPen(QPen(c))
        dot.setBrush(QBrush(c))
        
        dot.setData(0, "sam_point")
        
        self.scene_input.addItem(dot)
        
        self.view_input.update_point_scales()

        self.run_sam_inference(self.coordinates, self.labels)

    def handle_sam_box(self, rect):
        
        if self.is_busy():
            return # Interaction blocked during inference
    
        self.coordinates = [[rect.left(), rect.top()], [rect.right(), rect.bottom()]]
        self.labels = [2, 3]
        self.run_sam_inference(self.coordinates, self.labels)
        self.coordinates = []
        self.labels = []

    def _process_sam_points(self, coords, labels):
        """
        Helper to process input points/boxes for SAM inference.
        - Gets current viewport crop.
        - Filters points/boxes to be within the viewport.
        - Translates coordinates to be relative to the crop.
        - Returns crop, offsets, and valid coordinates, or None if no valid points.
        """
        crop, x_off, y_off = self.get_viewport_crop()
        if crop.width == 0 or crop.height == 0:
            return None

        view_rect = QRectF(x_off, y_off, crop.width, crop.height)
        valid_coords, valid_labels = [], []
        is_box = (labels == [2, 3])

        if is_box:
            box_rect = QRectF(QPointF(coords[0][0], coords[0][1]), QPointF(coords[1][0], coords[1][1])).normalized()
            if view_rect.intersects(box_rect):
                clipped_rect = box_rect.intersected(view_rect)
                valid_coords = [
                    [clipped_rect.left() - x_off, clipped_rect.top() - y_off],
                    [clipped_rect.right() - x_off, clipped_rect.bottom() - y_off]
                ]
                valid_labels = [2, 3]
        else:
            for (cx, cy), label in zip(coords, labels):
                if view_rect.contains(cx, cy):
                    valid_coords.append([cx - x_off, cy - y_off])
                    valid_labels.append(label)

        if not valid_coords or not valid_labels:
            self.model_output_mask = Image.new("L", self.original_image.size, 0)
            self.overlay_pixmap_item.setPixmap(QPixmap())
            self.status_label.setText("Idle (No points in view)")
            return None

        return crop, x_off, y_off, valid_coords, valid_labels


    def run_sam_inference(self, coords, labels):
        
        # Prevent multiple threads
        if self.is_busy():
            return
        
        model_name = self.combo_sam.currentText()
        if "Select" in model_name or "No Models" in model_name: return

        self.set_loading(True, "Running SAM Inference. First run on viewport is slow.")
        
        provider_data = self.combo_sam_model_EP.currentData() or ("CPUExecutionProvider", {}, "cpu")
        
        processed = self._process_sam_points(coords, labels)
        if not processed:
            self.clear_overlay()
            self.set_loading(False)
            return
        crop, x_off, y_off, valid_coords, valid_labels = processed
        current_crop_rect = (x_off, y_off, crop.width, crop.height)

        
        def _do_sam_work(model_manager, model_name, provider_data, crop, valid_coords, valid_labels, current_crop_rect):
            
            success, msg = model_manager.init_sam_session(model_name, provider_data)
            if not success: raise Exception(msg)
            
            prov_code = provider_data[2]
            if "sam2" in model_name:
                mask_arr, status = model_manager.run_sam2(crop, valid_coords, valid_labels, current_crop_rect, prov_code)
            else:
                mask_arr, status = model_manager.run_sam1(crop, valid_coords, valid_labels, current_crop_rect, prov_code)
            
            return {"mask": mask_arr, "status": status}


        self.worker = InferenceWorker(
            _do_sam_work, 
            self.model_manager, model_name, provider_data, 
            crop, valid_coords, valid_labels, current_crop_rect
        )
        
        self.worker.finished.connect(lambda res: self._on_inference_finished(res, x_off, y_off))
        self.worker.error.connect(lambda msg: (self.set_loading(False), QMessageBox.critical(self, "SAM Error", msg)))
        self.worker.start()

    
    def is_busy(self):
        """
        Checks if a background task is currently running.
        And a lazy quick check if any image is loaded to prevent models running on the placeholder blank image
        """
        is_running = hasattr(self, 'worker') and self.worker is not None and self.worker.isRunning()
        return is_running or not self.image_paths 

    def run_automatic_model(self, model_name=None):
        
        if self.is_busy():
            return # Prevent multiple threads
        
        # Check if run from a hotkey (u,i,o,b) or get name from model list
        if not model_name: 
            model_name = self.combo_whole.currentText()
        if "Select" in model_name or "No Models" in model_name:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("No Models Found")
            msg_box.setText("No models found. Please download models using the download manager or from the Github URL in the help box.")
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.addButton(QPushButton("OK"), QMessageBox.ButtonRole.AcceptRole)
            msg_box.exec()
            return

        crop, x_off, y_off = self.get_viewport_crop()
        if crop.width == 0 or crop.height == 0:
            return

        provider_data = self.combo_auto_model_EP.currentData()
        self.set_loading(True, f"Processing {model_name}...")

        def _do_auto_work(model_manager, model_name, provider_data, crop):
            prov_str, prov_opts, prov_code = provider_data
            session, load_t = model_manager.get_auto_session(model_name, provider_data)
            # Send prov_code to generate the status message
            mask_arr, status = model_manager.run_auto_inference(session, crop, model_name, load_t, prov_code)
            return {"mask": mask_arr, "status": status}

        self.worker = InferenceWorker(_do_auto_work, 
                                      self.model_manager, 
                                      model_name, 
                                      provider_data, 
                                      crop)
        self.worker.finished.connect(lambda res: self._on_inference_finished(res, x_off, y_off))
        self.worker.error.connect(lambda msg: (self.set_loading(False), QMessageBox.critical(self, "Error", msg)))
        self.worker.start()

    def _on_inference_finished(self, result, x_off, y_off):
        """Processes model output masks into a overlay, and updates UI"""
        mask_arr = result["mask"]
        status_msg = result["status"]

        # Paste the viewport mask into the correct place
        self.model_output_mask = Image.new("L", self.original_image.size, 0)
        self.model_output_mask.paste(Image.fromarray(mask_arr, mode="L"), (x_off, y_off))
        
        # Update UI
        self.show_mask_overlay()
        self.update_cached_model_icons()
        
        self.set_loading(False, status_msg)


    def show_mask_overlay(self):
        if self.model_output_mask:
            blue = Image.new("RGB", self.original_image.size, (0, 0, 255))
            overlay = blue.convert("RGBA")
            overlay.putalpha(self.model_output_mask)
            self.overlay_pixmap_item.setPixmap(pil2pixmap(overlay))
        
        self.update_trimap_preview()

    def clear_overlay(self):
        self.coordinates = []
        self.labels = []
        self.model_output_mask = Image.new("L", self.original_image.size, 0)
        self.overlay_pixmap_item.setPixmap(QPixmap())
        self.trimap_overlay_item.setPixmap(QPixmap())

        # Clear the invisible paint scratchpad
        if hasattr(self, 'paint_image'):
            self.paint_image.fill(Qt.GlobalColor.transparent)

        # Remove points/boxes AND the temporary paint path if it exists
        cursor_item = self.view_input.brush_cursor_item
        items_to_remove = []
        
        # Check if we have a temporary paint path lingering
        if hasattr(self, 'temp_path_item') and self.temp_path_item:
            items_to_remove.append(self.temp_path_item)
            self.temp_path_item = None

        for item in self.scene_input.items():
            # Don't delete the background image or the blue overlay
            if item == self.input_pixmap_item or item == self.overlay_pixmap_item: continue
            
            # Don't delete the cursor
            if item == cursor_item or item.parentItem() == cursor_item: continue
            
            # Delete SAM points/boxes
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem)):
                items_to_remove.append(item)
                
        for item in items_to_remove:
            self.scene_input.removeItem(item)

    def clear_visible_area(self):
        if not self.model_output_mask: return
        self.add_undo_step()
        crop, x, y = self.get_viewport_crop()
        vis_mask = Image.new("L", self.original_image.size, 0)
        draw = ImageDraw.Draw(vis_mask)
        draw.rectangle([x, y, x+crop.width, y+crop.height], fill=255)
        self.working_mask = ImageChops.subtract(self.working_mask, vis_mask)
        self.update_output_preview()

    def add_undo_step(self):
        self.undo_history.append(self.working_mask.copy())
        if len(self.undo_history) > UNDO_STEPS:
            self.undo_history.pop(0)
        if hasattr(self, "redo_history"):
            self.redo_history.clear() 


    def undo(self):
        if self.undo_history:
            # current state goes onto redo stack
            if hasattr(self, "redo_history"):
                self.redo_history.append(self.working_mask.copy())
            self.working_mask = self.undo_history.pop()
            self.update_output_preview()

    def redo(self):
        if hasattr(self, "redo_history") and self.redo_history:
            # current state goes back onto undo stack
            self.undo_history.append(self.working_mask.copy())
            if len(self.undo_history) > UNDO_STEPS:
                self.undo_history.pop(0)
            self.working_mask = self.redo_history.pop()
            self.update_output_preview()



    def add_mask(self): self.modify_mask(ImageChops.add)
    def subtract_mask(self): self.modify_mask(ImageChops.subtract)

    def on_trimap_mode_changed(self):
        """Shows or hides UI elements based on the selected trimap mode."""
        is_auto = self.rb_trimap_auto.isChecked()
        
        self.auto_trimap_sliders_widget.setVisible(is_auto)
        
        self.btn_edit_trimap.setVisible(not is_auto)
        
        self.update_trimap_preview()

    def open_trimap_editor(self):
        if not self.original_image:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        initial_trimap = None

        if hasattr(self, 'user_trimap') and self.user_trimap:
            initial_trimap = self.user_trimap
        # Otherwise, generate one from the current mask and sliders as a starting point.
        elif self.model_output_mask:
            fg_erode = self.sl_fg_erode.value()
            bg_erode = self.sl_bg_erode.value()
            trimap_np = self.model_manager.generate_trimap_from_mask(self.model_output_mask, fg_erode, bg_erode)
            initial_trimap = Image.fromarray(trimap_np)
        else:
            # If there's no mask at all, start with a blank (all unknown) trimap.
            initial_trimap = Image.new("L", self.original_image.size, 128)

        dialog = TrimapEditorDialog(self.original_image, initial_trimap, self)
        if dialog.exec():
            # If the user clicked OK, store the result
            self.user_trimap = dialog.final_trimap
            self.last_trimap = np.array(self.user_trimap) 
            
            self.rb_trimap_custom.setChecked(True)
            
            # Ensure the preview is shown
            self.chk_show_trimap.setChecked(True)
            self.status_label.setText("Custom trimap updated.")
        self.update_trimap_preview()

    def update_trimap_preview_throttled(self):
        """Starts a timer to update the trimap preview, preventing too many updates."""
        self.trimap_timer.start()
    
    def handle_alpha_matting_toggle(self, checked):
        """Manages UI state when the main Alpha Matting checkbox is toggled."""
        self.alpha_matting_frame.setVisible(checked)
        if checked:
            self.update_trimap_preview()
        else:
            # If unchecked, also uncheck the trimap view so it's off if re-enabled
            self.chk_show_trimap.setChecked(False)

    def toggle_trimap_display(self, checked):
        """Shows or hides the trimap overlay on the input view."""
        if checked:
            self.update_trimap_preview()
            self.trimap_overlay_item.show()
        else:
            self.trimap_overlay_item.hide()

    #@line_profiler.profile
    def update_trimap_preview(self):
        """Generates and displays the correct trimap based on the selected source."""
        if not self.model_output_mask or not self.chk_alpha_matting.isChecked() or not self.chk_show_trimap.isChecked():
            self.trimap_overlay_item.setPixmap(QPixmap())
            self.view_input.hide_legend()
            return

        trimap_np = None

        if self.rb_trimap_auto.isChecked():
            fg_erode = self.sl_fg_erode.value()
            bg_erode = self.sl_bg_erode.value()
            trimap_np = self.model_manager.generate_trimap_from_mask(self.model_output_mask, fg_erode, bg_erode)
        
        elif self.rb_trimap_custom.isChecked() and hasattr(self, 'user_trimap'):
            trimap_np = np.array(self.user_trimap)

        if trimap_np is not None:
            # Use a look up table for speed
            lut = np.zeros((256, 4), dtype=np.uint8)
            lut[0]   = [0, 0, 0, 255]       # Background -> Black
            lut[128] = [0, 0, 255, 255]     # Unknown -> Blue
            lut[255] = [255, 255, 255, 255] # Foreground -> White
            
            trimap_color = lut[trimap_np]
            self.trimap_overlay_item.setPixmap(numpy_to_pixmap(trimap_color))

            # Use fixed strings for the legend content
            legend_text = (
                "<b>Trimap Legend</b><br>"
                "⚪ White: Definite Foreground<br>"
                "🔵 Blue: Unknown (Alpha Edge)<br>"
                "⚫ Black: Definite Background"
            )
            self.view_input.show_legend(legend_text)
            
        else:
            self.trimap_overlay_item.setPixmap(QPixmap())
            self.view_input.hide_legend()

            

    def modify_mask(self, op):
        """
        Apply modifiers to the mask outputted by the models. Entire pipeline is threaded
        
        :param op: imagechops operation: add or subtract
        """
        
        if not self.model_output_mask: return
        if self.is_busy(): return # Prevent multiple threads

        self.add_undo_step()

        # Capture state for the worker thread 
        mask_to_process = self.model_output_mask.copy()
        apply_matting = self.chk_alpha_matting.isChecked()
        apply_soften = self.chk_soften.isChecked()
        apply_binary = self.chk_post.isChecked()

        msg = " Alpha matting can take a while" if apply_matting else ""
        self.set_loading(True, "Applying mask..." + msg)
        
        matting_params = {}
        if apply_matting:
            try:
                image_crop, x_off, y_off = self.get_viewport_crop()
                trimap_np = None

                # Get the correct trimap based on UI selection
                if self.rb_trimap_custom.isChecked() and hasattr(self, 'user_trimap'):
                    # If a custom trimap exists and is selected, use it.
                    trimap_crop_pil = self.user_trimap.crop((x_off, y_off, x_off + image_crop.width, y_off + image_crop.height))
                    trimap_np = np.array(trimap_crop_pil)
                else: 
                    # Fallback to the automatic generation method
                    mask_crop = mask_to_process.crop((x_off, y_off, x_off + image_crop.width, y_off + image_crop.height))
                    fg_erode = self.sl_fg_erode.value()
                    bg_erode = self.sl_bg_erode.value()
                    trimap_np = self.model_manager.generate_trimap_from_mask(mask_crop, fg_erode, bg_erode)
                
                matting_params = {
                    'image_crop': image_crop,
                    'trimap_np': trimap_np,
                    'x_off': x_off, 'y_off': y_off,
                    'algorithm': self.combo_matting_algorithm.currentText(),
                    # For now use the provider we have selected in automatic models combobox
                    # Unsure if worth giving user the option to select EP, since VitMatte is essentially a automatic model
                    'provider_data': self.combo_auto_model_EP.currentData()
                }
            except Exception as e:
                QMessageBox.critical(self, "Alpha Matting Prep Error", str(e))
                self.set_loading(False)
                return

        def _do_modify_work(model_manager, base_mask, matting_enabled, soften_enabled, binary_enabled, m_params):
            processed_mask = base_mask

            if matting_enabled and m_params:
                matted_alpha_crop = model_manager.run_matting(
                    m_params['algorithm'], 
                    m_params['image_crop'], 
                    m_params['trimap_np'], 
                    m_params['provider_data']
                )

                if matted_alpha_crop:
                    # Create a new mask to avoid modifying the original in the thread
                    new_m = processed_mask.copy()
                    # Create a black patch to clear the area under the new matted crop
                    paste_area = Image.new("L", matted_alpha_crop.size, 0)
                    new_m.paste(paste_area, (m_params['x_off'], m_params['y_off']))
                    # Paste the new matted result
                    new_m.paste(matted_alpha_crop, (m_params['x_off'], m_params['y_off']))
                    processed_mask = new_m

            if soften_enabled:
                processed_mask = processed_mask.filter(ImageFilter.GaussianBlur(radius=SOFTEN_RADIUS))
            
            if binary_enabled:
                arr = np.array(processed_mask) > 128
                kernel = np.ones((3,3), np.uint8)
                arr = cv2.erode(cv2.dilate(arr.astype(np.uint8), kernel, iterations=1), kernel, iterations=1).astype(bool)
                processed_mask = Image.fromarray((arr*255).astype(np.uint8))
            
            return processed_mask

        # Create and start the worker
        self.worker = InferenceWorker(
            _do_modify_work,
            self.model_manager, mask_to_process,
            apply_matting, apply_soften, apply_binary, matting_params
        )
        self.worker.finished.connect(lambda result_mask: self._on_modify_mask_finished(result_mask, op))
        self.worker.error.connect(lambda msg: (self.set_loading(False), QMessageBox.critical(self, "Mask Processing Error", msg)))
        self.worker.start()

    def _on_modify_mask_finished(self, processed_mask, op):
        """Handles the result from the mask modification worker."""
        self.working_mask = op(self.working_mask, processed_mask)
        self.set_loading(False, "Idle")
        self.update_output_preview()

    

    

    def toggle_shadow_options(self, checked):
        if checked: self.shadow_frame.show()
        else: self.shadow_frame.hide()
        self.update_output_preview()

    #@profile
    def render_output_image(self, shadow_downscale=0.125):
        if not self.original_image: return

        if self.chk_estimate_foreground.isChecked():
            
            try:
                current_mask_hash = hash(self.working_mask.tobytes())
                
                if current_mask_hash != self.working_mask_hash:

                    self.set_loading(True, "Estimating foreground colour correction")

                    cutout = self.model_manager.estimate_foreground(self.original_image, self.working_mask)

                    self.set_loading(False)
                    self.working_mask_hash = current_mask_hash
                    self.cached_fg_corrected_cutout = cutout
                else:
                    cutout = self.cached_fg_corrected_cutout

            except Exception as e:
                print(f"Error during foreground estimation: {e}")
                # Fallback to the standard method if something goes wrong
                cutout = self.original_image.convert("RGBA")
                cutout.putalpha(self.working_mask)
        else:
            # Quick enough to not need caching
            cutout = self.original_image.convert("RGBA")
            cutout.putalpha(self.working_mask)
        
        if self.chk_shadow.isChecked():
            # work in numpy for speed to avoid requiring caching
            op = self.sl_s_op.value()
            rad = self.sl_s_r.value()
            off_x, off_y = self.sl_s_x.value(), self.sl_s_y.value()
            
            w, h = self.working_mask.size
            
            m_np = np.array(self.working_mask)
            small_w = max(1, int(w * shadow_downscale))
            small_h = max(1, int(h * shadow_downscale))
            
            m_small = cv2.resize(m_np, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            
            blur_size = max(1, int(rad * shadow_downscale))

            m_blur_small = cv2.GaussianBlur(m_small, (0, 0), sigmaX=blur_size)
            m_blur_small = cv2.convertScaleAbs(m_blur_small, alpha=op/255.0)
            
            m_full = cv2.resize(m_blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Create a black background and use the shifted mask as Alpha
            shifted_alpha = np.zeros((h, w), dtype=np.uint8)
            
            # Calculate slice boundaries for the offset
            src_y1, src_y2 = max(0, -off_y), min(h, h - off_y)
            src_x1, src_x2 = max(0, -off_x), min(w, w - off_x)
            dst_y1, dst_y2 = max(0, off_y), min(h, h + off_y)
            dst_x1, dst_x2 = max(0, off_x), min(w, w + off_x)

            if dst_y2 > dst_y1 and dst_x2 > dst_x1:
                shifted_alpha[dst_y1:dst_y2, dst_x1:dst_x2] = m_full[src_y1:src_y2, src_x1:src_x2]
            
            shadow_layer_np = np.zeros((h, w, 4), dtype=np.uint8)
            shadow_layer_np[:, :, 3] = shifted_alpha
            cutout = Image.alpha_composite(Image.fromarray(shadow_layer_np), cutout)

        bg_txt = self.combo_bg_color.currentText()
        if bg_txt == "Transparent": 
            final = cutout
        elif "Blur" in bg_txt:

            mask_hash = hash(self.working_mask.tobytes())
            rad = getattr(self, 'blur_radius', 30)
            current_params = (mask_hash, rad)
            
            if current_params == self.last_blur_params and self.cached_blurred_bg is not None:
                final = self.cached_blurred_bg.copy()
            else:
                self.set_loading(True, "Blurring Background")
                
                orig_np = np.array(self.original_image)
                rgb = orig_np[:, :, :3]
                m_np = np.array(self.working_mask)

                # expand mask to reduce halo effects
                dilation_size = 7 
                kernel = np.ones((dilation_size, dilation_size), np.uint8)
                dilated_mask = cv2.dilate(m_np, kernel, iterations=1)
                
                # create weight map from the dilated mask
                # 1.0 = background (keep), 0.0 = dilated cutout (ignore)
                weight_map = (255 - dilated_mask).astype(np.float32) / 255.0
                          
                if rad % 2 == 0: rad += 1
                ksize = (rad, rad)

                # normalised convolution
                weighted_blur = cv2.blur(rgb * weight_map[..., None], ksize)
                
                blurred_weights = cv2.blur(weight_map, ksize)

                result = weighted_blur / (blurred_weights[..., None] + 1e-8)
                
                blur_final = cv2.convertScaleAbs(result)
                final = Image.fromarray(blur_final).convert("RGBA")
                
                final.alpha_composite(cutout)
                self.set_loading(False,"")

                self.cached_blurred_bg = final.copy()
                self.last_blur_params = current_params
                self.set_loading(False, "")

            final.alpha_composite(cutout)
        else:
            final = Image.new("RGBA", self.original_image.size, bg_txt.lower())
            final.alpha_composite(cutout)
        return final

    def update_output_preview(self):

        if self.chk_show_mask.isChecked(): 
            final = self.working_mask.convert("RGBA")
        else:
            final = self.render_output_image()
        
        self.output_pixmap_item.setPixmap(pil2pixmap(final))
        self.view_output.setSceneRect(self.view_input.sceneRect())

    def reset_all(self):
        self.clear_overlay()

        # Clear Trimap and reset UI
        self.last_trimap = None
        if hasattr(self, 'user_trimap'):
            del self.user_trimap
        self.rb_trimap_auto.setChecked(True)

        self.add_undo_step()
        self.working_mask = Image.new("L", self.original_image.size, 0)
        self.update_output_preview()
        
    def reset_working_image(self):
        self.add_undo_step()
        self.working_mask = Image.new("L", self.original_image.size, 0)
        self.update_output_preview()
        
    def copy_input_to_output(self):
        self.add_undo_step()
        self.working_mask = Image.new("L", self.original_image.size, 255)
        self.update_output_preview()

    def toggle_paint_mode(self, enabled):
        self.paint_mode = enabled
        self.view_input.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.view_output.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        
        # Update cursor on both views
        cursor_pos = self.view_input.mapFromGlobal(QCursor.pos())
        scene_pos = self.view_input.mapToScene(cursor_pos)
        self.view_input.update_brush_cursor(scene_pos)
        self.view_output.update_brush_cursor(scene_pos)

    def handle_paint_start(self, pos):
        
        if self.is_busy():
            return # Interaction blocked during inference
        
        # If a path is already being drawn, do nothing.
        if hasattr(self, 'current_path'): return

        buttons = QApplication.mouseButtons()
        self.is_erasing = bool(buttons & Qt.MouseButton.RightButton)
        
        zoom = self.view_input.transform().m11()
        if zoom == 0: zoom = 1
        
        # Calculate width in scene coordinates
        self.brush_width = PAINT_BRUSH_SCREEN_SIZE / zoom
        
        # Temporary vector path for performance
        self.current_path = QPainterPath()
        self.current_path.moveTo(pos)
        # Ensure a dot appears on single click
        self.current_path.lineTo(pos.x() + 0.001, pos.y()) 

        self.temp_path_item = QGraphicsPathItem()
        self.temp_path_item.setPath(self.current_path)
        
        # Red for Paint, White for Erase
        color = QColor(255, 0, 0, 150) if not self.is_erasing else QColor(255, 255, 255, 150)
        pen = QPen(color, self.brush_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.temp_path_item.setPen(pen)
        self.temp_path_item.setZValue(100) 
        
        self.scene_input.addItem(self.temp_path_item)

        self.temp_path_item_out = QGraphicsPathItem()
        self.temp_path_item_out.setPath(self.current_path)
        self.temp_path_item_out.setPen(pen) # Use same pen
        self.temp_path_item_out.setZValue(100)
        self.scene_output.addItem(self.temp_path_item_out)

    def handle_paint_move(self, last, curr):
        
        if self.is_busy():
            return # Interaction blocked during inference
        
        if hasattr(self, 'current_path'):
            self.current_path.lineTo(curr)
            self.temp_path_item.setPath(self.current_path)

            if hasattr(self, 'temp_path_item_out'):
                self.temp_path_item_out.setPath(self.current_path)

    def handle_paint_end(self):
        """
        Finalizes the manual brush stroke. 
        Directly edits the working mask for immediate drawing
        """
        if self.is_busy():
            return
        
        # Clean up visual items from both scenes
        if hasattr(self, 'temp_path_item'):
            self.scene_input.removeItem(self.temp_path_item)
            self.temp_path_item = None
        if hasattr(self, 'temp_path_item_out'):
            self.scene_output.removeItem(self.temp_path_item_out)
            self.temp_path_item_out = None
            
        if not hasattr(self, 'current_path'): 
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Setup the scratchpad QImage for the current stroke
            self.paint_image.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(self.paint_image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            
            # Draw the recorded path onto the scratchpad
            # White (255) as the mask value for the stroke
            pen = QPen(QColor(255, 255, 255, 255), self.brush_width, 
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(self.current_path)
            painter.end()
            
            # Convert Scratchpad to PIL Mask (Fast Buffer Access)
            ptr = self.paint_image.constBits()
            ptr.setsize(self.paint_image.sizeInBytes())
            h, w = self.paint_image.height(), self.paint_image.width()
            
            # Create NumPy view and extract mask channel
            arr = np.array(ptr, copy=False).reshape(h, w, 4)
            stroke_mask_np = arr[:, :, 0].copy()

            if self.chk_soften.isChecked():
                # ksize (0,0) allows OpenCV to compute the kernel size automatically from sigma
                stroke_mask_np = cv2.GaussianBlur(stroke_mask_np, (0, 0), sigmaX=SOFTEN_RADIUS)
            
            stroke_mask = Image.fromarray(stroke_mask_np, mode="L")

            self.add_undo_step()

            # Previously the app modified self.model_output_mask and used self.show_mask_overlay()
            # 
            # Changed to immediate editing of output image because can't think of a reason to
            # paintbrush the model output mask, which is added/sub anyway
            # When alpha matting, we can edit the trimap so editing the initial mask with paint 
            # probably limited benefit
            if self.is_erasing:
                # Right Click: Remove from composite
                self.working_mask = ImageChops.subtract(self.working_mask, stroke_mask)
            else:
                # Left Click: Add to composite (Paint from original image)
                self.working_mask = ImageChops.add(self.working_mask, stroke_mask)
            
            self.update_output_preview()
            
        finally:
            if hasattr(self, 'current_path'): 
                del self.current_path
            QApplication.restoreOverrideCursor()

    def open_image_editor(self):
        if not self.original_image: return
        
        reply = QMessageBox.question(self, "Edit Image", 
                                     "Editing the original image will reset the progress on your output image.\n\n"
                                     "Consider saving the current mask first. Continue?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            dlg = ImageEditorDialog(self, self.original_image)
            if dlg.exec():
                if dlg.final_image:
                    self.original_image = dlg.final_image.convert("RGBA")
                    self.init_working_buffers()
                    self.update_input_view()
                    self.update_output_preview()

    def quick_save_jpeg(self):
        if not self.original_image or not self.image_paths: return
        orig_path = self.image_paths[0] if self.image_paths[0] != "Clipboard" else "clipboard.png"
        path = os.path.splitext(orig_path)[0] + "_nobg.jpg"

        path = self._sanitise_filename_for_windows(path)

        fname, _ = QFileDialog.getSaveFileName(self, "Quick Save", path, "JPG (*.jpg)")
        if fname:

            fname = self._sanitise_filename_for_windows(fname)

            if not fname.lower().endswith(".jpg"): fname += ".jpg"
            empty = Image.new("RGBA", self.original_image.size, 0)
            cutout = Image.composite(self.original_image, empty, self.working_mask)
            final = Image.new("RGB", self.original_image.size, "white")
            final.paste(cutout, (0,0), cutout)
        
            save_params = {'quality': 95}
            if self.image_exif: save_params['exif'] = self.image_exif
        
            final.save(fname, **save_params)
            self.status_label.setText(f"Quick saved to {fname}")


    def save_image(self):
        if not self.image_paths: return
        
        dlg = SaveOptionsDialog(self)
        if not dlg.exec(): return
        data = dlg.get_data()
        
        fmt = data['format']
        ext_map = {"png": "png", "webp_lossless": "webp", "webp_lossy": "webp", "jpeg": "jpg"}
        default_ext = ext_map[fmt]
        
        initial_name = os.path.splitext(self.image_paths[0])[0] + "_nobg." + default_ext

        initial_name = self._sanitise_filename_for_windows(initial_name)

        fname, _ = QFileDialog.getSaveFileName(self, "Save Image", initial_name, f"{default_ext.upper()} (*.{default_ext})")
        if not fname: return

        fname = self._sanitise_filename_for_windows(fname)
        
        if not fname.lower().endswith(f".{default_ext}"): fname += f".{default_ext}"


        final_image = self.render_output_image()

        # Zero out RGB data in transparent areas for export.
        # This prevents "ghost backgrounds" when re-loading the file.
        clean_canvas = Image.new("RGBA", final_image.size, (0, 0, 0, 0))
        clean_canvas.paste(final_image, (0, 0), final_image)
        final_image = clean_canvas

        # If trimming is enabled, calculate the crop box and apply it.
        if data.get('trim', False):
            bbox = self.working_mask.getbbox()
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                
                # Expand bounding box to include shadow if enabled
                if self.chk_shadow.isChecked():
                    shadow_off_x, shadow_off_y = self.sl_s_x.value(), self.sl_s_y.value()
                    s_rad = self.sl_s_r.value()
                    s_min_x = min_x + shadow_off_x - s_rad
                    s_min_y = min_y + shadow_off_y - s_rad
                    s_max_x = max_x + shadow_off_x + s_rad
                    s_max_y = max_y + shadow_off_y + s_rad
                    
                    min_x = min(min_x, s_min_x)
                    min_y = min(min_y, s_min_y)
                    max_x = max(max_x, s_max_x)
                    max_y = max(max_y, s_max_y)

                # Crop the fully rendered image to the calculated bounding box
                # Clamp the calculated coordinates to the image boundaries
                orig_w, orig_h = final_image.size
                
                final_min_x = max(0, int(min_x))
                final_min_y = max(0, int(min_y))
                final_max_x = min(orig_w, int(max_x))
                final_max_y = min(orig_h, int(max_y))
                # Check for empty crop after clamping (e.g., if shadow only extends outside)
                if final_max_x > final_min_x and final_max_y > final_min_y:
                    # Apply the crop to the fully rendered image
                    final_image = final_image.crop((final_min_x, final_min_y, final_max_x, final_max_y))
                # Else: no change, keep the empty image (which is what render_output_image returns if mask is empty)

        if fmt == "jpeg":
            background = Image.new("RGB", final_image.size, (255, 255, 255))
            # Ensure the image has an alpha channel to use as a mask
            if final_image.mode != 'RGBA':
                final_image = final_image.convert('RGBA')
            background.paste(final_image, mask=final_image.split()[3])
            final_image = background
        
        save_params = {}
        if self.image_exif: save_params['exif'] = self.image_exif
        if fmt == "jpeg": save_params['quality'] = data['quality']
        elif fmt == "webp_lossy": save_params['quality'] = data['quality']
        elif fmt == "webp_lossless": save_params['lossless'] = True
        elif fmt == "png": save_params['optimize'] = True

        try:
            final_image.save(fname, **save_params)
            self.status_label.setText(f"Saved to {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            
        if data['save_mask']:
            mname = os.path.splitext(fname)[0] + "_mask.png"
            self.working_mask.save(mname)
            self.status_label.setText(f"Saved to {os.path.basename(fname)} and {os.path.basename(mname)}")

    def show_help(self):
        d = QDialog(self)
        d.setWindowTitle("Help / About")
        d.resize(600, 500)
        l = QVBoxLayout(d)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setText("""Interactive Background Remover by Sean (Prickly Gorse)

https://github.com/pricklygorse/Interactive-Image-Background-Remover

Load an image, then use either a Automatic background removal model (e.g. u2net, isnet, BiRefNet) or Interactive Segment Anything (Left Click/Drag Box).
Working mask appears as blue overlay on Input.
Press 'A' to Add mask to Output, 'S' to Subtract.

Image segmentation models are run on the current view, so you can zoom into details to fine tune your background removal.

Controls:
- Left Click: Add Point (SAM) / Start Box
- Right Click: Add Negative Point
- Middle Click: Pan
- Scroll: Zoom 
- Ctrl: Zoom with touchpad
- P: Toggle Paintbrush (draw manually on mask). Right click - Add, Left click - Erase

Shortcuts:
- A: Add current mask to output
- S: Subtract current mask from output
- C: Clear overlay points
- W: Reset output image
- R: Reset everything
- U: Run u2net
- I: Run isnet
- O: Run rmbg1_4
- B: Run ben2_base
- Ctrl+S: Save As
- Ctrl+Shift+S: Quick Save JPG (White BG)
- Ctrl+Z: Undo
- Ctrl+Y: Redo
        """)
        l.addWidget(t)
        d.exec()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Image Background Remover")
    parser.add_argument("images", nargs="*", help="Paths to images")
    
    parser.add_argument("--alpha-matting", action="store_true", help="Start with Alpha Matting enabled")
    parser.add_argument("--bg-colour", type=str)
    parser.add_argument("--binary", action="store_true", help="Start with Binary Mask enabled")
    parser.add_argument("--soften", action="store_true", help="Start with Soften Mask enabled")
    parser.add_argument("--shadow", action="store_true", help="Start with Drop Shadow enabled")
    parser.add_argument("--colour-correction", action="store_true", help="Start with Colour Correction enabled")
    parser.add_argument("--load-mask", action="store_true", help="Load associated _mask.png file if present")



    args = parser.parse_args()

    app = QApplication(sys.argv)
    # Set fusion for consistency across OS. The theme will be loaded from settings.
    app.setStyle("Fusion")
    window = BackgroundRemoverGUI(args.images, args)
    window.showMaximized()
    sys.exit(app.exec())