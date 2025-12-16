#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import cv2
import gc
from timeit import default_timer as timer
#import line_profiler

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QSplitter, QDialog, QScrollArea, 
                             QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem, 
                             QTextEdit, QSizePolicy, QRadioButton, QButtonGroup, QInputDialog, 
                             QProgressBar, QStyle)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (QPixmap, QImage, QColor, QPainter, QPainterPath, QPen, QBrush,
                         QKeySequence, QShortcut, QCursor, QIcon, QPalette)

from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageGrab, ImageFilter, ImageChops
import onnxruntime as ort


from pymatting import estimate_alpha_cf, estimate_foreground_ml


import src.download_manager as download_manager
from src.ui_widgets import CollapsibleFrame, SynchronisedGraphicsView
from src.ui_dialogs import SaveOptionsDialog, ImageEditorDialog
from src.trimap_editor import TrimapEditorDialog
from src.utils import pil2pixmap, numpy_to_pixmap
from src.constants import PAINT_BRUSH_SCREEN_SIZE, UNDO_STEPS, SOFTEN_RADIUS, SAM_TRT_WARMUP_POINTS


if getattr(sys, 'frozen', False):
    SCRIPT_BASE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ROOT_DIR = os.path.join(SCRIPT_BASE_DIR, "Models/")

CACHE_ROOT_DIR = os.path.join(SCRIPT_BASE_DIR, "Models", "cache")




# --- Helper Functions ---

def get_available_ep_options():
        """
        Returns list of (Display Name, ProviderStr, OptionsDict, ShortCode)
        """
        try:
            available = ort.get_available_providers()
            print(available)
        except:
            print("No onnxruntime providers")
            return []
        options = []
        
        # DEBUG override to show all providers
        #available = ["CUDAExecutionProvider", "CPUExecutionProvider", "TensorrtExecutionProvider", "OpenVINOExecutionProvider", "CoreMLExecutionProvider"] # DEBUG
        
        if "TensorrtExecutionProvider" in available:
            # We will generate specific cache paths at runtime, 
            # so we pass an empty dict here, or default options.
            options.append(("TensorRT (GPU)", "TensorrtExecutionProvider", {}, "trt"))

        if "CUDAExecutionProvider" in available:
            options.append(("CUDA (GPU)", "CUDAExecutionProvider", {}, "cuda"))

        # Windows generic provider
        if "DmlExecutionProvider" in available:
            options.append(("DirectML (GPU)", "DmlExecutionProvider", {}, "dml"))

        if "OpenVINOExecutionProvider" in available:
            try:
                ov_devices = ort.capi._pybind_state.get_available_openvino_device_ids()
            except Exception:
                ov_devices = []
            
            # Fallback if query fails but provider exists
            if not ov_devices: 
                ov_devices = ['CPU']

            for dev in ov_devices:
                label = f"OpenVINO-{dev}"
                opts = {'device_type': dev} 
                options.append((label, "OpenVINOExecutionProvider", opts, f"ov-{dev.lower()}"))

        # MAC
        if "CoreMLExecutionProvider" in available:
            options.append(("CoreML", "CoreMLExecutionProvider", {}, "coreml"))

        # CPU always available
        options.append(("CPU", "CPUExecutionProvider", {}, "cpu"))

        return options


AVAILABLE_EPS = get_available_ep_options()



# --- Main Application ---

class BackgroundRemoverGUI(QMainWindow):
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths if image_paths else []
        self.current_image_index = 0
        self.setWindowTitle("Interactive Image Background Remover")
        self.resize(1600, 900)

        self.coordinates = []
        self.labels = []
        self.undo_history = []
        self.redo_history = []
        self.paint_mode = False
        self.sam_model_path = None
        self.image_exif = None
        self.blur_radius = 30

        # Track TensorRT warmup per (model, provider) combo
        self._sam_trt_warmed = set()

        # Persistent settings
        self.settings = QSettings("PricklyGorse", "InteractiveBackgroundRemover")

        self.original_image = None
        self.working_image = None
        self.working_mask = None
        self.last_trimap = None
        self.loaded_whole_models = {}
        self.loaded_sam_models = {}
        self.loaded_matting_models = {} 
        self.model_output_mask = None

        self.init_ui()
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
        
        saved_theme = self.settings.value("theme", "light")
        self.set_theme(saved_theme)


    def init_ui(self):
        self.setAcceptDrops(True)
        main = QWidget(); self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # --- Sidebar ---
        sidebar_container = QFrame()
        sidebar_container.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        scroll_widget = QWidget()
        sl = QVBoxLayout(scroll_widget) # Main layout for controls
        scroll_widget.setFixedWidth(sidebar_container.width()) # Constrain content width
        scroll_area.setWidget(scroll_widget)
        sidebar_layout.addWidget(scroll_area)

        nav = QHBoxLayout()
        btn_open_file = QPushButton("Open"); btn_open_file.clicked.connect(self.load_image_dialog)
        btn_open_clipboard = QPushButton("Clipboard"); btn_open_clipboard.clicked.connect(self.load_clipboard)
        self.btn_prev_image = QPushButton("<<")
        self.btn_prev_image.setMaximumWidth(40)
        self.btn_prev_image.setToolTip("Load Previous Image (Left Arrow Key)")
        self.btn_prev_image.clicked.connect(self.load_previous_image)
        self.btn_next_image = QPushButton(">>")
        self.btn_next_image.setMaximumWidth(40)
        self.btn_next_image.setToolTip("Load Next Image (Right Arrow Key)")
        self.btn_next_image.clicked.connect(self.load_next_image)
        nav.addWidget(btn_open_file); nav.addWidget(btn_open_clipboard); nav.addWidget(self.btn_prev_image); nav.addWidget(self.btn_next_image)
        sl.addLayout(nav)

        self.update_prev_next_button_state()
        
        btn_edit_image = QPushButton("Edit Image"); btn_edit_image.clicked.connect(self.open_image_editor)
        btn_load_mask = QPushButton("Load Mask"); btn_load_mask.clicked.connect(self.load_mask_dialog)

        h_edit_load_buttons = QHBoxLayout()
        h_edit_load_buttons.addWidget(btn_edit_image)
        h_edit_load_buttons.addWidget(btn_load_mask)
        sl.addLayout(h_edit_load_buttons)


        self.available_eps = AVAILABLE_EPS

        # --- Hardware Acceleration Dropdown ---
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
        sl.addWidget(self.hw_options_frame)
        last_auto_cache_mode = self.settings.value("auto_ram_cache_mode", 1, type=int)
        self.auto_cache_group.blockSignals(True)
        self.auto_cache_group.button(last_auto_cache_mode).setChecked(True)
        self.auto_cache_group.blockSignals(False)
        self.trt_cache_option_visibility() # Initial check for TensorRT

        # Rest of UI
        h_models_header = QHBoxLayout()
        lbl_models = QLabel("<b> Models:</b>")
        lbl_models.setContentsMargins(3, 0, 0, 0)
        h_models_header.addWidget(lbl_models)
        h_models_header.addStretch()

        self.btn_download = QPushButton("Download 📥")
        self.btn_download.setToolTip("Download Models...")
        self.btn_download.setFixedSize(120, 32)
        self.btn_download.clicked.connect(self.open_download_manager)
        h_models_header.addWidget(self.btn_download)
        sl.addLayout(h_models_header)
        
        lbl_sam = QLabel(" Interactive (SAM):")
        lbl_sam.setToolTip("<b>Segment Anything Models</b><br>"
                           "These require you to interact with the image.<br>"
                           "<i>Usage: Left-click to add points, right-click to add negative (avoid) points, or drag to draw boxes around the subject.</i><br><br>"
                           "Disc drive icons show models that have saved optimised versions cached.")
        sl.addWidget(lbl_sam)

        self.combo_sam = QComboBox()
        self.combo_sam.setToolTip(lbl_sam.toolTip())
        self.populate_sam_models()
        sl.addWidget(self.combo_sam)
        lbl_whole = QLabel(" Automatic (Whole Image):")
        lbl_whole.setToolTip("<b>Automatic Models</b><br>"
                             "These run automatically on the entire image.<br>"
                             "<i>Usage: Select a model and click 'Run Automatic'. No points needed.</i><br><br>"
                             "Disc drive icons show models that have saved optimised versions cached.")
        sl.addWidget(lbl_whole)

        # Whole Image Combo
        self.combo_whole = QComboBox()
        self.combo_whole.setToolTip(lbl_whole.toolTip()) # Reuse the tooltip
        self.populate_whole_models()
        sl.addWidget(self.combo_whole)
        
        btn_whole = QPushButton("Run Automatic"); btn_whole.clicked.connect(lambda: self.run_automatic_model())
        sl.addWidget(btn_whole)

        lbl_actions = QLabel("<b> Actions:</b>")
        lbl_actions.setContentsMargins(3, 0, 0, 0)
        sl.addWidget(lbl_actions)
        h_act = QHBoxLayout()
        btn_add = QPushButton("Add Mask"); btn_add.clicked.connect(self.add_mask)
        btn_sub = QPushButton("Sub Mask"); btn_sub.clicked.connect(self.subtract_mask)
        h_act.addWidget(btn_add); h_act.addWidget(btn_sub)
        sl.addLayout(h_act)
        
        h_ut = QHBoxLayout()
        btn_undo = QPushButton("Undo"); btn_undo.clicked.connect(self.undo)
        btn_clr = QPushButton("Clear Points/Masks"); btn_clr.clicked.connect(self.clear_overlay)
        h_ut.addWidget(btn_undo); h_ut.addWidget(btn_clr)
        sl.addLayout(h_ut)
        
        h_rs = QHBoxLayout()
        btn_rst = QPushButton("Reset Img"); btn_rst.clicked.connect(self.reset_working_image)
        btn_all = QPushButton("Reset All"); btn_all.clicked.connect(self.reset_all)
        h_rs.addWidget(btn_rst); h_rs.addWidget(btn_all)
        sl.addLayout(h_rs)
        
        h_vs = QHBoxLayout()
        btn_cp = QPushButton("Copy In->Out"); btn_cp.clicked.connect(self.copy_input_to_output)
        btn_c_vis = QPushButton("Clear Visible"); btn_c_vis.clicked.connect(self.clear_visible_area)
        h_vs.addWidget(btn_cp); h_vs.addWidget(btn_c_vis) 
        sl.addLayout(h_vs)

        lbl_options = QLabel("<b>Options:</b>")
        lbl_options.setContentsMargins(3, 0, 0, 0)
        sl.addWidget(lbl_options)

        bg_layout = QHBoxLayout()
        bg_label = QLabel("Background:")
        bg_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        bg_layout.addWidget(bg_label)
        self.combo_bg_color = QComboBox()
        colors = ["Transparent", "White", "Black", "Red", "Blue", 
                  "Orange", "Yellow", "Green", "Grey", 
                  "Lightgrey", "Brown", "Blurred (Slow)"]
        self.combo_bg_color.addItems(colors)
        self.combo_bg_color.currentTextChanged.connect(self.handle_bg_change)
        bg_layout.addWidget(self.combo_bg_color)
        sl.addLayout(bg_layout)

        self.chk_paint = QCheckBox("Paintbrush (P)"); self.chk_paint.toggled.connect(self.toggle_paint_mode)
        sl.addWidget(self.chk_paint)
        
        self.chk_show_mask = QCheckBox("Show Mask"); self.chk_show_mask.toggled.connect(self.update_output_preview)
        sl.addWidget(self.chk_show_mask)
        
        self.chk_post = QCheckBox("Binary Mask (no partial transparency)")
        sl.addWidget(self.chk_post)
        
        self.chk_soften = QCheckBox("Soften Mask/Paintbrush Edges")
        soften_checked = self.settings.value("soften_mask", False, type=bool)
        self.chk_soften.setChecked(soften_checked)
        self.chk_soften.toggled.connect(lambda checked: self.settings.setValue("soften_mask", checked))
        sl.addWidget(self.chk_soften)
        
        self.chk_alpha_matting = QCheckBox("Alpha Matting (Slow, Experimental)")
        self.chk_alpha_matting.setToolTip("Uses a matting algorithm to estimate the transparency of mask edges.\n"
                                          "This can improve the quality of detailed edges such as hair, especially when using binary mask models like SAM.\n"
                                          "This requires a trimap, either estimated from a SAM or automatic models, or manually drawn.\n"
                                          "This is computationally expensive and is only applied when 'Add' or 'Subtract' is clicked. Undo if the effect is unsatisfactory")
        sl.addWidget(self.chk_alpha_matting)

        self.chk_alpha_matting.toggled.connect(self.handle_alpha_matting_toggle)
        sl.addWidget(self.chk_alpha_matting)

        self.chk_alpha_matting.toggled.connect(self.handle_alpha_matting_toggle)
        sl.addWidget(self.chk_alpha_matting)

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


        sl.addWidget(self.alpha_matting_frame)
        self.alpha_matting_frame.hide()

        self.chk_estimate_foreground = QCheckBox("Mask Edge Colour Correction (Slow)")
        self.chk_estimate_foreground.setToolTip("Recalculates edge colors to remove halos or fringes from the original background.\n"
                                                "Recommended for soft edges such as hair")
        sl.addWidget(self.chk_estimate_foreground)
        self.chk_estimate_foreground.toggled.connect(self.update_output_preview)


        self.chk_shadow = QCheckBox("Drop Shadow")
        self.chk_shadow.toggled.connect(self.toggle_shadow_options)
        sl.addWidget(self.chk_shadow)

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
            
        sl.addWidget(self.shadow_frame)
        self.shadow_frame.hide()

        sl.addStretch()
        btn_save = QPushButton("Save As..."); btn_save.clicked.connect(self.save_image)
        sl.addWidget(btn_save)
        btn_qsave = QPushButton("Quick Save JPG"); btn_qsave.clicked.connect(self.quick_save_jpeg)
        sl.addWidget(btn_qsave)

        bottom_buttons_layout = QHBoxLayout()

        btn_help = QPushButton("Help / About"); btn_help.clicked.connect(self.show_help)
        bottom_buttons_layout.addWidget(btn_help)

        self.btn_theme_toggle = QPushButton("🔆")
        self.btn_theme_toggle.setToolTip("Toggle Dark/Light Mode")
        self.btn_theme_toggle.clicked.connect(self.toggle_light_dark_mode)
        self.btn_theme_toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        bottom_buttons_layout.addWidget(self.btn_theme_toggle)
        sl.addLayout(bottom_buttons_layout)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        
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
        self.splitter.addWidget(w_in)
        
        w_out = QWidget(); l_out = QVBoxLayout(w_out)
        w_out.setMinimumWidth(150)

        # Output header with split orientation change button
        h_out_header = QHBoxLayout()
        h_out_header.addWidget(QLabel("Output Composite"))
        h_out_header.addStretch()
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
        self.splitter.addWidget(w_out)

        self.view_input.set_sibling(self.view_output)
        self.view_output.set_sibling(self.view_input)

        layout.addWidget(sidebar_container)
        layout.addWidget(self.splitter, 1) # Give splitter stretch factor and add it to the main layout

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

            self.btn_theme_toggle.setText("🌙")
            self.btn_theme_toggle.setToolTip("Switch to Light Mode")
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
            
            self.btn_theme_toggle.setText("🔆")
            self.btn_theme_toggle.setToolTip("Switch to Dark Mode")
            self.settings.setValue("theme", "light")

        # Tweaks for collapsible menu and splitter toggle button
        self.hw_options_frame.collapsible_set_light_dark()
        self.toggle_splitter_orientation(initial_setup=True)

    def toggle_light_dark_mode(self):
        """Switches from the current theme to the other."""
        app = QApplication.instance()
        if not app: return

        current_bg = app.palette().color(QPalette.ColorRole.Window)
        if current_bg.lightness() < 128:
            self.set_theme('light')
        else:
            self.set_theme('dark')

    def _warmup_sam_trt(self, max_points=SAM_TRT_WARMUP_POINTS):
        """
        Warm up the SAM decoder on TensorRT by running it once with `max_points`
        synthetic points so the engine is compiled for that max shape.

        After that, any run with <= max_points points should reuse this engine.
        """
        model_name = self.combo_sam.currentText()
        if "sam2" in model_name:
            self._warmup_sam2_trt(max_points)
        else:
            self._warmup_sam1_trt(max_points)


    def _warmup_sam1_trt(self, max_points):
        """
        Warmup for classic SAM / mobile_sam path used by run_sam_inference().

        We don't need a real image here; TensorRT only cares about shapes.
        So we feed dummy zeros with the same shapes as the real pipeline.
        """
        if not hasattr(self, "sam_encoder") or not hasattr(self, "sam_decoder"):
            return

        input_size = (684, 1024)  # (H, W)
        h, w = input_size

        # Encoder
        enc_input_name = self.sam_encoder.get_inputs()[0].name
        dummy_img = np.zeros((h, w, 3), dtype=np.float32)

        encoder_inputs = {enc_input_name: dummy_img}
        embedding = self.sam_encoder.run(None, encoder_inputs)[0]

        # --- Decoder warmup: single call with max_points (+1 sentinel) ---
        n = max_points

        onnx_coord = np.zeros((1, n + 1, 2), dtype=np.float32)
        onnx_label = np.zeros((1, n + 1), dtype=np.float32)
        onnx_label[0, :n] = 1.0 
        onnx_label[0, -1] = -1.0  # sentinel

        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)
        orig_im_size = np.array(input_size, dtype=np.float32)  # shape (2,)

        candidate_inputs = {
            "image_embeddings": embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": mask_input,
            "has_mask_input": has_mask_input,
            "orig_im_size": orig_im_size,
        }

        decoder_input_names = [d.name for d in self.sam_decoder.get_inputs()]
        run_inputs = {
            name: candidate_inputs[name]
            for name in decoder_input_names
            if name in candidate_inputs
        }

        # One shot at max_points – this should build the TensorRT engine
        self.sam_decoder.run(None, run_inputs)



    def _warmup_sam2_trt(self, max_points):
        """
        Warmup for SAM2 path used by run_samv2_inference().
        """
        if not hasattr(self, "sam_encoder") or not hasattr(self, "sam_decoder"):
            return

        enc_inputs = self.sam_encoder.get_inputs()
        enc_input_h, enc_input_w = enc_inputs[0].shape[2:]

        # Encoder
        image_rgb = np.zeros((enc_input_h, enc_input_w, 3), dtype=np.float32)
        input_tensor = image_rgb.transpose(2, 0, 1)[np.newaxis, :, :, :]
        encoder_inputs = {enc_inputs[0].name: input_tensor}
        high_res_feats_0, high_res_feats_1, image_embed = self.sam_encoder.run(None, encoder_inputs)

        # Decoder. Collect inputs to accomodate Vietdev or Ibaigorodo exported models
        orig_h, orig_w = enc_input_h, enc_input_w
        scale_factor = 4
        mask_input = np.zeros(
            (1, 1, enc_input_h // scale_factor, enc_input_w // scale_factor),
            dtype=np.float32
        )
        has_mask_input = np.array([0], dtype=np.float32)
        original_size_np = np.array([orig_h, orig_w], dtype=np.int32)

        all_base_inputs = {
            'image_embed': image_embed,
            'high_res_feats_0': high_res_feats_0,
            'high_res_feats_1': high_res_feats_1,
            'mask_input': mask_input,
            'has_mask_input': has_mask_input,
            'orig_im_size': original_size_np
        }

        decoder_input_names = [d.name for d in self.sam_decoder.get_inputs()]

        n = max_points
        points = np.zeros((1, n, 2), dtype=np.float32)
        labels = np.ones((1, n), dtype=np.float32)  # all positive

        candidate_inputs = dict(all_base_inputs)
        candidate_inputs["point_coords"] = points
        candidate_inputs["point_labels"] = labels

        run_inputs = {
            name: candidate_inputs[name]
            for name in decoder_input_names
            if name in candidate_inputs
        }

        self.sam_decoder.run(None, run_inputs)


    def toggle_splitter_orientation(self, initial_setup=False):
        current_orientation = self.splitter.orientation()
        
        if initial_setup:
            target_orientation = current_orientation
        else:
            target_orientation = Qt.Orientation.Vertical if current_orientation == Qt.Orientation.Horizontal else Qt.Orientation.Horizontal
            self.splitter.setOrientation(target_orientation)

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

    
    
    def check_is_cached(self, model_name, provider_short_code):
        """
        Checks if a model has cache files on disk. 
        We rely on the directory naming convention from _create_inference_session.
        """
        if not os.path.isdir(CACHE_ROOT_DIR):
            return False

        if provider_short_code == 'cpu': return False

        # We need to reconstruct the folder name logic roughly. 
        # Since short_code maps to specific provider/device combos, we can filter Models/cache
                
        # Simple heuristic: Look for folder containing model_name and provider code
        # This allows "trt" to match "TensorrtExecutionProvider_u2net"
        
        sanitised_model_name = "".join([c for c in model_name if c.isalnum() or c in "-_"])
        
        for folder in os.listdir(CACHE_ROOT_DIR):
            full_path = os.path.join(CACHE_ROOT_DIR, folder)
            if not os.path.isdir(full_path): continue
            
            if sanitised_model_name in folder and len(os.listdir(full_path)) > 0:
                # Distinguish between providers
                if provider_short_code == 'trt' and 'Tensorrt' in folder: return True
                if provider_short_code.startswith('ov') and 'OpenVINO' in folder:
                    # Check specific device match (e.g. ov-gpu)
                    device = provider_short_code.split('-')[1].upper()
                    if device in folder: return True
                
        return False
    
    def _get_cached_icon(self):
        """
        Returns a standard system icon to indicate a model is cached on disk.
        Returns a cached QIcon to avoid fetching it every time.
        """
        if hasattr(self, "_cached_drive_icon"):
            return self._cached_drive_icon

        # Fetch the standard "Hard Drive" icon from the current application style
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DriveHDIcon)
        
        self._cached_drive_icon = icon
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
                if self.check_is_cached(m_name, short_code):
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
                if self.check_is_cached(m_name, sam_code):
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

        # Old session is now invalid and needs removing
        if self.loaded_whole_models:
            print("Execution provider changed, clearing model cache.")
            for key, session in self.loaded_whole_models.items():
                del session
            self.loaded_whole_models.clear()
            gc.collect() 

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
            if self.loaded_whole_models:
                print("Auto model memory cache option changed, clearing cache.")
                self.loaded_whole_models.clear()
                gc.collect()
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

    def on_sam_cache_changed(self, button, checked):
        if checked:
            # Clear loaded models regardless of option chosen
            # to allow easy unloading and consistent behaviour
            if self.loaded_sam_models:
                print("SAM RAM Cache option changed, clearing cache.")
                self.loaded_sam_models.clear()

            if hasattr(self, 'sam_encoder'): del self.sam_encoder
            if hasattr(self, 'sam_decoder'): del self.sam_decoder
            self.sam_model_path = None
            if hasattr(self, "encoder_output"): delattr(self, "encoder_output")
            gc.collect()
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

    def on_sam_EP_changed(self, index):
        """
        Handle change in SAM execution provider.
        """
        # Retrieve from combobox userdata (ProviderStr, OptionsDict, ShortCode)
        data = self.combo_sam_model_EP.itemData(index)
        if not data: return
        prov_str, prov_opts, short_code = data

        self.settings.setValue("sam_exec_short_code", short_code)

        # Old session is now invalid and needs removing
        if hasattr(self, 'sam_encoder'): del self.sam_encoder
        if hasattr(self, 'sam_decoder'): del self.sam_decoder
        
        self.sam_model_path = None 
        if hasattr(self, "encoder_output"): delattr(self, "encoder_output")

        if self.loaded_sam_models:
            self.loaded_sam_models.clear()

        gc.collect()

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
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents() 
        else:
            self.progress_bar.hide()
            self.status_label.setText(message if message else "Idle")
            self.status_label.setStyleSheet("")
            QApplication.restoreOverrideCursor()

    def showEvent(self, event):
        super().showEvent(event)
        w = self.splitter.width()
        self.splitter.setSizes([w//2, w//2])

    def open_download_manager(self):
        dlg = download_manager.ModelDownloadDialog(
            model_root_dir=MODEL_ROOT_DIR, 
            main_app_instance=self, 
            parent=self
        )
        dlg.exec()

    def populate_sam_models(self):
        sam_models = ["mobile_sam", "sam_vit_b", "sam_vit_h", "sam_vit_l", "sam2"]
        matches = []
        if os.path.exists(MODEL_ROOT_DIR):
            for filename in os.listdir(MODEL_ROOT_DIR):
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
        whole_models = ["rmbg", "isnet", "u2net", "BiRefNet"]
        matches = []
        if os.path.exists(MODEL_ROOT_DIR):
            for filename in os.listdir(MODEL_ROOT_DIR):
                for partial in whole_models:
                    if partial in filename and ".onnx" in filename:
                        matches.append(filename.replace(".onnx",""))
        self.combo_whole.clear()
        if matches: self.combo_whole.addItems(sorted(list(set(matches))))
        else: self.combo_whole.addItem("No Models Found")

    def populate_matting_models(self):
        self.combo_matting_algorithm.clear()
        # PyMatting is always an option
        self.combo_matting_algorithm.addItem("PyMatting (CPU)")

        # Scan for ViTMatte (and in the future, more models)
        matting_models = ["vitmatte"]
        if os.path.exists(MODEL_ROOT_DIR):
            for filename in os.listdir(MODEL_ROOT_DIR):
                for partial in matting_models:
                    if partial in filename and filename.endswith(".onnx"):
                        model_name = filename.replace(".onnx", "")
                        self.combo_matting_algorithm.addItem(model_name)

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
        QShortcut(QKeySequence(Qt.Key.Key_Left), self).activated.connect(self.load_previous_image) # Previous Image
        QShortcut(QKeySequence(Qt.Key.Key_Right), self).activated.connect(self.load_next_image) # Next Image
        
        QShortcut(QKeySequence("U"), self).activated.connect(lambda: self.run_automatic_model("u2net"))
        QShortcut(QKeySequence("I"), self).activated.connect(lambda: self.run_automatic_model("isnet"))
        QShortcut(QKeySequence("O"), self).activated.connect(lambda: self.run_automatic_model("rmbg"))
        QShortcut(QKeySequence("B"), self).activated.connect(lambda: self.run_automatic_model("BiRefNet"))

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
            self.update_prev_next_button_state()
            self.update_window_title()
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

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
            self.status_label.setText("Loaded from Clipboard [1/1]")
            self.update_window_title()
            if hasattr(self, 'update_next_button_state'):
                self.update_prev_next_button_state()
            return

        QMessageBox.information(self, "Clipboard Empty", 
                                "No image or valid image path found on the clipboard.")

    def load_image_dialog(self):
        file_filter = "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.PNG *.JPG *.JPEG *.WEBP *.BMP *.TIF *.TIFF);;All Files (*)"
        fnames, _ = QFileDialog.getOpenFileNames(self, "Open", "", file_filter)
        if fnames:
            self.image_paths = fnames
            self.current_image_index = 0
            self.load_image(fnames[0])
            if hasattr(self, 'update_next_button_state'):
                self.update_prev_next_button_state()

    def load_mask_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Mask", "", "Images (*.png)")
        if fname:
            mask = Image.open(fname).convert("L")
            if mask.size == self.original_image.size:
                self.add_undo_step()
                self.working_mask = mask
                self.update_output_preview()

    def load_next_image(self):
        if self.image_paths and self.image_paths[0] != "Clipboard" and len(self.image_paths) > 1:
            if self.current_image_index < len(self.image_paths) - 1:
                self.current_image_index += 1
                self.load_image(self.image_paths[self.current_image_index])

    def load_previous_image(self):
        if self.image_paths and self.image_paths[0] != "Clipboard" and len(self.image_paths) > 1:
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.load_image(self.image_paths[self.current_image_index])

    

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

                QApplication.processEvents()

                self._init_sam()
                
                self.set_loading(False,f"Pre-loaded SAM: {self.combo_sam.currentText()}")

            else:
                print(f"Startup SAM model '{sam_model_id}' not found.")

        if auto_model_id:
            idx = self.combo_whole.findText(auto_model_id, Qt.MatchFlag.MatchContains)
            if idx >= 0:
                self.combo_whole.setCurrentIndex(idx)
                QApplication.processEvents()
                
                # This logic is adapted from run_automatic_model to only load the session
                model_name = self.combo_whole.currentText()
                model_path = os.path.join(MODEL_ROOT_DIR, model_name + ".onnx")
                prov_str, prov_opts, prov_code = self.combo_auto_model_EP.currentData()
                cache_key = f"{model_name}_{prov_code}"

                if cache_key not in self.loaded_whole_models:
                    try:
                        self.set_loading(True,f"Pre-loading automatic model: {model_name}")
                        #self.status_label.setStyleSheet("color: red;")
                        QApplication.processEvents()
                        session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
                        self.loaded_whole_models[cache_key] = session
                        self.set_loading(False,f"Pre-loaded Auto: {model_name}")
                        #self.status_label.setStyleSheet("")
                    except Exception as e:
                        print(f"Failed to pre-load automatic model '{model_name}': {e}")

    def update_prev_next_button_state(self):
        if not self.image_paths:
            self.btn_next_image.setEnabled(False)
            self.btn_prev_image.setEnabled(False)
            return

        is_clipboard = (self.image_paths[0] == "Clipboard")
        
        self.btn_next_image.setEnabled(self.current_image_index < len(self.image_paths) - 1 and not is_clipboard)
        self.btn_prev_image.setEnabled(self.current_image_index > 0 and not is_clipboard)

        
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
        
        self.last_trimap = None
        if hasattr(self, 'user_trimap'):
            del self.user_trimap

        if hasattr(self, "encoder_output"): 
            delattr(self, "encoder_output")
        self.last_crop_rect = None 
        
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

    def _init_sam(self):
        model_name = self.combo_sam.currentText()
        if "Select" in model_name or "No Models" in model_name:
            return False

        model_path = os.path.join(MODEL_ROOT_DIR, model_name)
        
        # Check if the currently active model is already the one we want
        if self.sam_model_path == model_path and hasattr(self, 'sam_encoder'):
            return True

        # Retrieve provider data from the SAM combobox
        # Data format: (ProviderStr, OptionsDict, ShortCode)
        data = self.combo_sam_model_EP.currentData()
        if not data:
            # Fallback if something is wrong
            prov_str, prov_opts, prov_code = ("CPUExecutionProvider", {}, "cpu")
        else:
            prov_str, prov_opts, prov_code = data

        cache_key = f"{model_name}_{prov_code}"
        cache_mode = self.sam_cache_group.checkedId()
        
        if cache_mode > 0 and cache_key in self.loaded_sam_models:
            self.sam_encoder, self.sam_decoder = self.loaded_sam_models[cache_key]
            self.sam_model_path = model_path
            if hasattr(self, "encoder_output"): delattr(self, "encoder_output")
            self.status_label.setText(f"SAM Ready (Cached RAM): {model_name}")
            return True

        # Not in cache, load
        self.set_loading(True, f"Loading SAM ({model_name}) on {prov_code}...")

        try:
            # Pass 'model_name' so encoder and decoder share the same disc cache folder (openvino, tensorRT) 
            encoder_sess = self._create_inference_session(
                model_path + ".encoder.onnx", 
                prov_str, 
                prov_opts, 
                model_name # Cache ID
            )
            decoder_sess = self._create_inference_session(
                model_path + ".decoder.onnx", 
                prov_str, 
                prov_opts, 
                model_name
            )

            if cache_mode == 1: # Keep last
                self.loaded_sam_models.clear() 
                gc.collect()
                self.loaded_sam_models[cache_key] = (encoder_sess, decoder_sess)
            elif cache_mode == 2: # Keep all
                self.loaded_sam_models[cache_key] = (encoder_sess, decoder_sess)

            # Set active references
            self.sam_encoder = encoder_sess
            self.sam_decoder = decoder_sess
            self.sam_model_path = model_path
            
            if hasattr(self, "encoder_output"):
                delattr(self, "encoder_output")


            # --- TensorRT warmup: pre-build engine for up to N points ---
            if prov_str == "TensorrtExecutionProvider":
                warm_key = f"{model_name}_{prov_code}"
                if warm_key not in self._sam_trt_warmed:
                    self.status_label.setText(
                        f"Warming up SAM TensorRT engine ({model_name}) "
                        f"for up to {SAM_TRT_WARMUP_POINTS} points..."
                    )
                    QApplication.processEvents()
                    try:
                        self._warmup_sam_trt(max_points=SAM_TRT_WARMUP_POINTS)
                    except Exception as e:
                        print(f"SAM TensorRT warmup failed: {e}")
                    else:
                        self._sam_trt_warmed.add(warm_key)



            self.set_loading(False, f"SAM Ready: {model_name} ({prov_code})")
            
            self.update_cached_model_icons()
            return True

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load SAM model:\n{e}")
            self.sam_model_path = None
            return False

    def handle_sam_point(self, scene_pos, is_positive):
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
        self.coordinates = [[rect.left(), rect.top()], [rect.right(), rect.bottom()]]
        self.labels = [2, 3]
        self.run_sam_inference(self.coordinates, self.labels)
        self.coordinates = []
        self.labels = []


    def _create_inference_session(self, model_path, provider_str, provider_options, model_id_name):
        """
        Generic builder for ONNX sessions. 
        Handles cache directory creation automatically based on provider + model name.
        """

        # Make path absolute so ONNX external data is resolved from the model's folder,
        # not from the process working directory.
        model_path = os.path.abspath(model_path)

        # These session options stop VRAM/RAM usage ballooning with subsequent model runs
        # by disabling automatic memory allocation
        # Doesn't seem to affect performance
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        
        final_providers = []
        cache_dir = None
        
        if provider_str == "TensorrtExecutionProvider":
            sub_dir_name = f"{provider_str}_{model_id_name}"
            if 'device_type' in provider_options:
                sub_dir_name = f"{provider_str}-{provider_options['device_type']}_{model_id_name}"
            sub_dir_name = "".join([c for c in sub_dir_name if c.isalnum() or c in "-_"])
            cache_dir = os.path.join(CACHE_ROOT_DIR, sub_dir_name)
            os.makedirs(cache_dir, exist_ok=True)

            trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
            }
            trt_opts.update(provider_options) # Merge any other opts
            final_providers.append((provider_str, trt_opts))

        elif provider_str == "OpenVINOExecutionProvider":

            sub_dir_name = f"{provider_str}_{model_id_name}"
            if 'device_type' in provider_options:
                sub_dir_name = f"{provider_str}-{provider_options['device_type']}_{model_id_name}"
            sub_dir_name = "".join([c for c in sub_dir_name if c.isalnum() or c in "-_"])
            cache_dir = os.path.join(CACHE_ROOT_DIR, sub_dir_name)
            os.makedirs(cache_dir, exist_ok=True)

            # Inject Cache Config
            ov_opts = {
                "cache_dir": cache_dir,
                "num_streams": 1, # Often helps stability
            }
            ov_opts.update(provider_options) # Adds 'device_type': 'GPU' etc.
            final_providers.append((provider_str, ov_opts))

        else:

            final_providers.append((provider_str, provider_options))

        # Always add CPU as fallback
        if provider_str != "CPUExecutionProvider":
            final_providers.append("CPUExecutionProvider")

        return ort.InferenceSession(model_path, sess_options=sess_options, providers=final_providers)

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
        valid_coords = []
        valid_labels = []

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


    def run_samv2_inference(self, coords, labels):
        """
        Slightly different logic for SAM2
        Depending on model download source (to save exporting myself), decoders vary. 
        Vietdev models don't resize the masks to original image size
        so we have to resize. Ibaigorodo's models include the resizer, so processing has to be ambigious.
        """
        if not self._init_sam(): return

        processed = self._process_sam_points(coords, labels)
        if not processed:
            return
        crop, x_off, y_off, valid_coords, valid_labels = processed
        
        orig_h, orig_w = crop.height, crop.width
        current_crop_rect = (x_off, y_off, crop.width, crop.height)

        final_status = "Idle"

        self.set_loading(True, "Running SAM Encoder...")
        QApplication.processEvents()
        try:
            encoder_time = 0.0
            decoder_time = 0.0

            # --- 1. Encoder ---
            if not hasattr(self, "encoder_output") or getattr(self, "last_crop_rect", None) != current_crop_rect:
                
                t_start = timer()

                enc_inputs = self.sam_encoder.get_inputs()
                enc_input_h, enc_input_w = enc_inputs[0].shape[2:]

                image_rgb = np.array(crop.convert("RGB"))
                input_img = cv2.resize(image_rgb, (enc_input_w, enc_input_h))
                mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                input_img = (input_img / 255.0 - mean) / std
                input_tensor = input_img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

                encoder_inputs = {enc_inputs[0].name: input_tensor}
                self.encoder_output = self.sam_encoder.run(None, encoder_inputs)

                self.last_crop_rect = current_crop_rect
                self.last_enc_shape = (enc_input_h, enc_input_w)
                encoder_time = (timer() - t_start) * 1000

            # --- 2. Decoder ---
            t_start = timer()
            high_res_feats_0, high_res_feats_1, image_embed = self.encoder_output
            enc_input_h, enc_input_w = self.last_enc_shape

            points = np.array(valid_coords, dtype=np.float32)[np.newaxis, ...]
            labels_np = np.array(valid_labels, dtype=np.float32)[np.newaxis, ...]
            points[..., 0] = points[..., 0] / orig_w * enc_input_w
            points[..., 1] = points[..., 1] / orig_h * enc_input_h

            scale_factor = 4
            mask_input = np.zeros((1, 1, enc_input_h // scale_factor, enc_input_w // scale_factor), dtype=np.float32)
            has_mask_input = np.array([0], dtype=np.float32)
            original_size_np = np.array([orig_h, orig_w], dtype=np.int32)

            all_possible_inputs = {
                'image_embed': image_embed, 'high_res_feats_0': high_res_feats_0, 'high_res_feats_1': high_res_feats_1,
                'point_coords': points, 'point_labels': labels_np, 'mask_input': mask_input,
                'has_mask_input': has_mask_input, 'orig_im_size': original_size_np
            }

            decoder_input_names = [d.name for d in self.sam_decoder.get_inputs()]
            decoder_inputs = {name: all_possible_inputs[name] for name in decoder_input_names if name in all_possible_inputs}

            dec_outputs = self.sam_decoder.run(None, decoder_inputs)
            masks, scores = dec_outputs[0], dec_outputs[1]
            decoder_time = (timer() - t_start) * 1000

            # --- 3. Post-processing ---
            scores, masks = scores.squeeze(), masks.squeeze()
            best_mask_idx = 0 if scores.ndim == 0 else np.argmax(scores)
            best_mask = masks if masks.ndim == 2 else masks[best_mask_idx]

            final_mask = cv2.resize(best_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            binary_mask = (final_mask > 0.0).astype(np.uint8) * 255

            self.model_output_mask = Image.new("L", self.original_image.size, 0)
            self.model_output_mask.paste(Image.fromarray(binary_mask, mode="L"), (x_off, y_off))
            self.show_mask_overlay()

            if encoder_time > 0:
                final_status = f"SAM: Encoder {encoder_time:.0f}ms | Decoder {decoder_time:.0f}ms"
            else:
                final_status = f"SAM: Encoder (Cached) | Decoder {decoder_time:.0f}ms"

        except Exception as e:
            print(f"SAM Error: {e}")
            final_status = "SAM Error"
        finally:
            QApplication.restoreOverrideCursor()
            self.set_loading(False, final_status)

    def run_sam_inference(self, coords, labels):
        
        if "sam2" in self.combo_sam.currentText():
            self.run_samv2_inference(coords, labels)
            return
        
        if not self._init_sam(): return
        
        processed = self._process_sam_points(coords, labels)
        if not processed:
            return
        crop, x_off, y_off, valid_coords, valid_labels = processed

        current_crop_rect = (x_off, y_off, crop.width, crop.height)
        final_status = "Idle"
        self.set_loading(True,"Running Sam Encoder...")
        QApplication.processEvents() 
        try:
            target_size, input_size = 1024, (684, 1024)
            img_np = np.array(crop.convert("RGB"))
            
            encoder_time = 0.0
            decoder_time = 0.0
            
            # --- Encoder ---
            if not hasattr(self, "encoder_output") or getattr(self, "last_crop_rect", None) != current_crop_rect:
                #self.status_label.setText("Running SAM Encoder...")
                #QApplication.processEvents() 

                t_start = timer()
                
                scale = min(input_size[1] / img_np.shape[1], input_size[0] / img_np.shape[0])
                transform_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
                cv_image = cv2.warpAffine(img_np, transform_matrix[:2], (input_size[1], input_size[0]), flags=cv2.INTER_LINEAR)
                
                encoder_inputs = {self.sam_encoder.get_inputs()[0].name: cv_image.astype(np.float32)}
                self.encoder_output = self.sam_encoder.run(None, encoder_inputs)
                
                self.last_crop_rect = current_crop_rect
                self.last_transform = transform_matrix
                self.last_orig_size = img_np.shape[:2]
                
                encoder_time = (timer() - t_start) * 1000

            # --- Decoder ---
            t_start = timer()
            
            embedding = self.encoder_output[0]
            onnx_coord = np.concatenate([np.array(valid_coords), np.array([[0.0, 0.0]])], axis=0)[None, :, :]
            onnx_label = np.concatenate([np.array(valid_labels), np.array([-1])], axis=0)[None, :].astype(np.float32)
            coords_aug = np.concatenate([onnx_coord, np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32)], axis=2)
            onnx_coord = np.matmul(coords_aug, self.last_transform.T)[:, :, :2].astype(np.float32)

            decoder_inputs = {
                "image_embeddings": embedding,
                "point_coords": onnx_coord, "point_labels": onnx_label,
                "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                "has_mask_input": np.zeros(1, dtype=np.float32),
                "orig_im_size": np.array(input_size, dtype=np.float32),
            }
            masks = self.sam_decoder.run(None, decoder_inputs)[0]
            
            inv_mtx = np.linalg.inv(self.last_transform)
            mask_resized = cv2.warpAffine(masks[0, 0, :, :], inv_mtx[:2], (self.last_orig_size[1], self.last_orig_size[0]), flags=cv2.INTER_LINEAR)
            mask_final = (mask_resized > 0.0).astype(np.uint8) * 255
            
            decoder_time = (timer() - t_start) * 1000
            
            self.model_output_mask = Image.new("L", self.original_image.size, 0)
            self.model_output_mask.paste(Image.fromarray(mask_final, mode="L"), (x_off, y_off))
            self.show_mask_overlay()

            if encoder_time > 0:
                final_status = f"SAM: Encoder {encoder_time:.0f}ms | Decoder {decoder_time:.0f}ms"
            else:
                final_status = f"SAM: Encoder (Cached) | Decoder {decoder_time:.0f}ms"

        except Exception as e: 
            print(f"SAM Error: {e}")
            final_status = "SAM Error"
        finally:
            QApplication.restoreOverrideCursor()
            self.set_loading(False, final_status)

    def run_automatic_model(self, model_name=None):
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


        found_name = None
        for f in os.listdir(MODEL_ROOT_DIR):
            if model_name in f and f.endswith(".onnx"):
                found_name = f.replace(".onnx", "")
                break
        if not found_name:
            self.status_label.setText(f"Model {model_name} not found")
            return

        model_name = found_name
        model_path = os.path.join(MODEL_ROOT_DIR, model_name + ".onnx")
        crop, x_off, y_off = self.get_viewport_crop()
        if crop.width == 0 or crop.height == 0:
            return

        prov_str, prov_opts, prov_code = self.combo_auto_model_EP.currentData()

        session = None
        cache_key = f"{model_name}_{prov_code}"
        load_time = 0.0 
        cache_mode = self.auto_cache_group.checkedId()

        if cache_mode > 0 and cache_key in self.loaded_whole_models:
            session = self.loaded_whole_models[cache_key]

        if session is None:
            self.set_loading(True, f"Loading and running {model_name} on {prov_code.upper()}...")
            t_start = timer()
            try:
                if cache_mode == 1: # Cache Last
                    # Clear previous models before adding the new one
                    if self.loaded_whole_models:
                        for key, sess in self.loaded_whole_models.items():
                            del sess
                        self.loaded_whole_models.clear()
                        gc.collect()
                
                session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
                load_time = (timer() - t_start) * 1000
                self.update_cached_model_icons()

                self.loaded_whole_models[cache_key] = session

            except Exception as e:
                self.set_loading(False, "Model load failed")
                QMessageBox.critical(self, "Model Load Error", f"Failed to create ONNX session for {model_name} on {prov_code.upper()}:\n\n{e}")
                return
        
        # Only set loading once if the session was already cached
        # otherwise loading cursor persists after inference 
        if load_time == 0.0:
            self.set_loading(True, f"Running {model_name}...")
        
        # Get model input size from the session
        try:
            input_shape = session.get_inputs()[0].shape
            # Assume shape is [batch, channels, height, width]
            target_h, target_w = input_shape[2], input_shape[3]

            if "rmbg" in model_name.lower() and "2" in model_name.lower():
                # rmbg2 needs specifying manually
                target_h = 1024
                target_w = 1024

        except Exception as e:
            self.set_loading(False, "Error reading model input.")
            QMessageBox.critical(
                self,
                "Model Error",
                f"Could not read valid H/W from the model input shape ({session.get_inputs()[0].shape}):\n{e}"
            )
            return

        final_status = "Idle"
        inference_time = 0.0
        try:
            # --- Preprocessing ---
            img_r = crop.convert("RGB").resize((target_w, target_h), Image.BICUBIC)

            if "isnet" in model_name or "rmbg" in model_name:
                mean, std = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
            else:
                mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

            im = np.array(img_r) / 255.0
            tmp = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
            tmp[:, :, 0] = (im[:, :, 0] - mean[0]) / std[0]
            tmp[:, :, 1] = (im[:, :, 1] - mean[1]) / std[1]
            tmp[:, :, 2] = (im[:, :, 2] - mean[2]) / std[2]

            inp = np.expand_dims(tmp.transpose((2, 0, 1)), 0).astype(np.float32)

            # --- Inference ---
            t_start_inf = timer()
            result = session.run(None, {session.get_inputs()[0].name: inp})[0]
            inference_time = (timer() - t_start_inf) * 1000

            # --- Postprocessing ---
            mask = result[0][0]
            
            if "BiRefNet" in model_name:
                mask = 1 / (1 + np.exp(-mask))

            denom = (mask.max() - mask.min()) or 1.0
            mask = (mask - mask.min()) / denom

            res_mask = Image.fromarray((mask * 255).astype("uint8"), "L").resize(
                crop.size, Image.Resampling.LANCZOS
            )
            self.model_output_mask = Image.new("L", self.original_image.size, 0)
            self.model_output_mask.paste(res_mask, (x_off, y_off))
            self.show_mask_overlay()

            load_str = f"{load_time:.0f}ms" if load_time > 0 else "Cached"
            final_status = f"{model_name} ({prov_code.upper()}): Load {load_str} | Inf {inference_time:.0f}ms"

            if cache_mode == 0: # No Caching
                del session
                gc.collect()
        except Exception as e: 
            QMessageBox.critical(self, "Error", str(e))
            final_status = "Inference Error"
        finally:
            self.set_loading(False, final_status)

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
            trimap_np = self.generate_trimap_from_mask(self.model_output_mask, fg_erode, bg_erode)
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
            self.trimap_overlay_item.setPixmap(QPixmap()) # Clear if not needed
            return

        trimap_np = None

        if self.rb_trimap_auto.isChecked():
            fg_erode = self.sl_fg_erode.value()
            bg_erode = self.sl_bg_erode.value()
            trimap_np = self.generate_trimap_from_mask(self.model_output_mask, fg_erode, bg_erode)
        
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
        else:
            self.trimap_overlay_item.setPixmap(QPixmap()) # Clear if no valid trimap

    def modify_mask(self, op):
        if not self.model_output_mask: return
        self.add_undo_step()
        m = self.model_output_mask

        if self.chk_alpha_matting.isChecked():
            self.set_loading(True, "Applying Alpha Matting...")
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
                    mask_crop = m.crop((x_off, y_off, x_off + image_crop.width, y_off + image_crop.height))
                    fg_erode = self.sl_fg_erode.value()
                    bg_erode = self.sl_bg_erode.value()
                    trimap_np = self.generate_trimap_from_mask(mask_crop, fg_erode, bg_erode)
                
                selected_algorithm = self.combo_matting_algorithm.currentText()
                
                if "vitmatte" in selected_algorithm.lower():
                    matted_alpha_crop = self.run_vitmatte_inference(image_crop, trimap_np)
                else: # Default to PyMatting
                    matted_alpha_crop = self.run_pymatting(image_crop, trimap_np)

                if matted_alpha_crop:
                    new_m = m.copy()
                    paste_area = Image.new("L", matted_alpha_crop.size, 0)
                    new_m.paste(paste_area, (x_off, y_off))
                    new_m.paste(matted_alpha_crop, (x_off, y_off))
                    m = new_m
            except Exception as e:
                QMessageBox.critical(self, "Alpha Matting Error", f"An error occurred during alpha matting:\n{e}")
            finally:
                self.set_loading(False, "Idle")
        
        if self.chk_soften.isChecked():
            m = m.filter(ImageFilter.GaussianBlur(radius=SOFTEN_RADIUS))
            
        if self.chk_post.isChecked():
            arr = np.array(m) > 128
            kernel = np.ones((3,3), np.uint8)
            arr = cv2.erode(cv2.dilate(arr.astype(np.uint8), kernel, iterations=1), kernel, iterations=1).astype(bool)
            m = Image.fromarray((arr*255).astype(np.uint8))
        self.working_mask = op(self.working_mask, m)
        self.update_output_preview()

    def generate_trimap_from_mask(self, mask_pil, fg_erode_size, bg_erode_size):
        """
        Generates a three-tone trimap from a binary mask using erosion.
        Returns the trimap as a NumPy array (0=BG, 128=Unknown, 255=FG).
        """
        mask_np = np.array(mask_pil)
        foreground_threshold = 240
        background_threshold = 10

        is_foreground = mask_np > foreground_threshold
        is_background = mask_np < background_threshold

        # Erode foreground
        if fg_erode_size > 0:
            fg_kernel = np.ones((fg_erode_size, fg_erode_size), np.uint8)
            is_foreground_eroded = cv2.erode(is_foreground.astype(np.uint8), fg_kernel, iterations=1)
        else:
            is_foreground_eroded = is_foreground

        # Erode background
        if bg_erode_size > 0:
            bg_kernel = np.ones((bg_erode_size, bg_erode_size), np.uint8)
            is_background_eroded = cv2.erode(is_background.astype(np.uint8), bg_kernel, iterations=1)
        else:
            is_background_eroded = is_background

        trimap = np.full(mask_np.shape, dtype=np.uint8, fill_value=128)
        trimap[is_foreground_eroded.astype(bool)] = 255
        trimap[is_background_eroded.astype(bool)] = 0
        
        return trimap

    def run_pymatting(self, image_crop_pil, trimap_np):
        """
        Calculates the alpha matte using the PyMatting library.
        Returns the alpha matte as a PIL Image.
        """
        img_normalized = np.array(image_crop_pil.convert("RGB")) / 255.0
        trimap_normalized = trimap_np / 255.0

        alpha = estimate_alpha_cf(img_normalized, trimap_normalized)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(alpha, mode="L")

    def run_vitmatte_inference(self, image_crop_pil, trimap_np):
        """
        Runs inference with a ViTMatte ONNX model.
        Returns the alpha matte as a PIL Image.
        """
        model_name = self.combo_matting_algorithm.currentText()
        prov_str, prov_opts, prov_code = self.combo_auto_model_EP.currentData()
        cache_key = f"{model_name}_{prov_code}"
        session = self.loaded_matting_models.get(cache_key)

        # TODO - respect users caching options
        if session is None:
            model_path = os.path.join(MODEL_ROOT_DIR, model_name + ".onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ViTMatte model not found at: {model_path}")
            
            self.set_loading(True, f"Loading {model_name}...")
            session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
            self.loaded_matting_models[cache_key] = session
            self.set_loading(False)

        # Preprocess
        original_size = image_crop_pil.size
        trimap_pil = Image.fromarray(trimap_np)
        
        # use 1024 for performance, but any multiple of 32 will work
        # TODO maybe let user choose size. On GPU no reason to restrict users
        target_size = (1024,1024)

        image_resized = image_crop_pil.convert("RGB").resize(target_size, Image.BILINEAR)
        trimap_resized = trimap_pil.convert("L").resize(target_size, Image.NEAREST)

        image_np = (np.array(image_resized, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        trimap_np_resized = np.expand_dims(np.array(trimap_resized, dtype=np.float32) / 255.0, axis=-1)
        
        combined = np.concatenate((image_np, trimap_np_resized), axis=2)
        combined = np.transpose(combined, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        # Inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: combined})

        # --- Postprocess ---
        alpha = outputs[0][0][0]
        alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode='L')
        alpha_image = alpha_image.resize(original_size, Image.LANCZOS)
        
        return alpha_image


    def toggle_shadow_options(self, checked):
        if checked: self.shadow_frame.show()
        else: self.shadow_frame.hide()
        self.update_output_preview()

    def render_output_image(self, shadow_downscale=0.125):
        if not self.original_image: return
        
        if self.chk_estimate_foreground.isChecked():
            try:
                image_rgb_normalized = np.array(self.original_image.convert("RGB")) / 255.0
                alpha_normalized = np.array(self.working_mask.convert("L")) / 255.0

                self.set_loading(True, "Estimating foreground colour correction")
                foreground_rgb_normalized = estimate_foreground_ml(image_rgb_normalized, alpha_normalized)

                foreground_rgb = np.clip(foreground_rgb_normalized * 255, 0, 255)
                alpha_channel = np.clip(alpha_normalized * 255, 0, 255)
                
                cutout_array = np.dstack((foreground_rgb, alpha_channel)).astype(np.uint8)
                
                cutout = Image.fromarray(cutout_array, "RGBA")

                self.set_loading(False)

            except Exception as e:
                print(f"Error during foreground estimation: {e}")
                # Fallback to the standard method if something goes wrong
                empty = Image.new("RGBA", self.original_image.size, 0)
                cutout = Image.composite(self.original_image, empty, self.working_mask)
        
        else:

            empty = Image.new("RGBA", self.original_image.size, 0)
            cutout = Image.composite(self.original_image, empty, self.working_mask)
        
        if self.chk_shadow.isChecked():
            op = self.sl_s_op.value()
            rad = self.sl_s_r.value()
            off_x, off_y = self.sl_s_x.value(), self.sl_s_y.value()
            
            orig_size = self.working_mask.size
            
            # Process shadow at a much lower resolution for performance. This feature is very slow
            small_w = max(1, int(orig_size[0] * shadow_downscale))
            small_h = max(1, int(orig_size[1] * shadow_downscale))
            
            shadow_small = self.working_mask.resize((small_w, small_h), Image.Resampling.NEAREST)
            
            scaled_rad = rad * shadow_downscale
            shadow_small = shadow_small.filter(ImageFilter.GaussianBlur(scaled_rad))
            
            shadow = shadow_small.resize(orig_size, Image.Resampling.BICUBIC)
            
            shadow = shadow.point(lambda p: int(p * (op/255)))
            
            sl = Image.new("RGBA", self.original_image.size, 0)
            sl.putalpha(shadow)
            bg_layer = Image.new("RGBA", self.original_image.size, 0)
            bg_layer.paste(sl, (off_x, off_y), sl)
            cutout = Image.alpha_composite(bg_layer, cutout)

        bg_txt = self.combo_bg_color.currentText()
        if bg_txt == "Transparent": 
            final = cutout
        elif "Blur" in bg_txt:
            m_np = np.array(self.working_mask)
            k = np.ones((5,5),np.uint8)
            d_mask = cv2.dilate(m_np, k, iterations=1)
            orig_rgb = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGBA2RGB)
            f_bg = cv2.inpaint(orig_rgb, d_mask, 3, cv2.INPAINT_TELEA)
            rad = getattr(self, 'blur_radius', 30)
            if rad % 2 == 0: rad += 1
            blur = cv2.blur(f_bg, (rad, rad))
            final = Image.fromarray(blur).convert("RGBA")
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
        
        if hasattr(self, 'current_path'):
            self.current_path.lineTo(curr)
            self.temp_path_item.setPath(self.current_path)

            if hasattr(self, 'temp_path_item_out'):
                self.temp_path_item_out.setPath(self.current_path)

    def handle_paint_end(self):
      
        # Clean up visual item
        if hasattr(self, 'temp_path_item'):
            self.scene_input.removeItem(self.temp_path_item)
            self.temp_path_item = None
        if hasattr(self, 'temp_path_item_out'):
            self.scene_output.removeItem(self.temp_path_item_out)
            self.temp_path_item_out = None
            
        if not hasattr(self, 'current_path'): return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Setup the scratchpad QImage
            self.paint_image.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(self.paint_image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            
            # Draw the recorded path onto the scratchpad
            # White because we just want the alpha shape
            pen = QPen(QColor(255, 255, 255, 255), self.brush_width, 
                      Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawPath(self.current_path)
            painter.end()
            
            # Convert Scratchpad to PIL Mask (Fast Buffer Access)
            ptr = self.paint_image.constBits()
            ptr.setsize(self.paint_image.sizeInBytes())
            h, w = self.paint_image.height(), self.paint_image.width()
            
            # Zero-copy view into the QImage data
            arr = np.array(ptr, copy=False).reshape(h, w, 4)
            # Extract the Blue channel (or Alpha) as our mask
            stroke_mask = Image.fromarray(arr[:, :, 0].copy(), mode="L")
            
            # Apply to Model Output
            if self.is_erasing:
                self.model_output_mask = ImageChops.subtract(self.model_output_mask, stroke_mask)
            else:
                self.model_output_mask = ImageChops.add(self.model_output_mask, stroke_mask)
            
            self.show_mask_overlay()
            
        finally:
            if hasattr(self, 'current_path'): del self.current_path
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
- B: Run BiRefNet
- Ctrl+S: Save As
- Ctrl+Shift+S: Quick Save JPG (White BG)
- Ctrl+Z: Undo
- Ctrl+Y: Redo
        """)
        l.addWidget(t)
        d.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Set fusion for consistency across OS. The theme will be loaded from settings.
    app.setStyle("Fusion")
    window = BackgroundRemoverGUI(sys.argv[1:])
    window.showMaximized()
    sys.exit(app.exec())