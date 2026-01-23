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
                             QProgressBar, QStyle, QToolBar, QTabWidget, QSpacerItem,QColorDialog)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QPropertyAnimation, QEasingCurve, QThread, pyqtSignal, QEvent, QMimeData, QBuffer, QIODevice
from PyQt6.QtGui import (QPixmap, QImage, QColor, QPainter, QPainterPath, QPen, QBrush,
                         QKeySequence, QShortcut, QCursor, QIcon, QPalette, QAction, QGuiApplication)

from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageGrab, ImageFilter, ImageChops



try: pyi_splash.update_text("Loading App Scripts")
except: pass

import src.settings_download_manager as settings_download_manager
from src.ui_styles import apply_theme
from src.ui_widgets import CollapsibleFrame, SynchronisedGraphicsView, ThumbnailList, OrientationSplitter
from src.ui_dialogs import InpaintingDialog
from src.trimap_editor import TrimapEditorDialog
from src.utils import pil2pixmap, numpy_to_pixmap, apply_tone_sharpness, generate_drop_shadow, \
    generate_blurred_background, sanitise_filename_for_windows, get_current_crop_bbox, generate_trimap_from_mask, clean_alpha, generate_alpha_map, \
    generate_outline, generate_inner_glow, apply_subject_tint, compose_final_image, refine_mask
from src.constants import PAINT_BRUSH_SCREEN_SIZE, UNDO_STEPS, SOFTEN_RADIUS

try: pyi_splash.update_text("Loading pymatting (Compiles on first run, approx 1-2 minutes)")
except: pass

print("Loading pymatting. On first run this will take a minute or two as it compiles")
from src.model_manager import ModelManager 

VIEW_IN_MSG = "Models run on current view. Zoom for more detail.   L-Click: Add Point | R-Click: Add Avoid Point | Drag: Box | M-Click: Pan | Scroll: Zoom (Ctrl+Scroll Touchpad) | [P]: Paintbrush"
VIEW_OUT_MSG = "OUTPUT | M-Click: Pan | Scroll: Zoom | [A/S]: Add/Subtract Current Model Mask from Output"
VIEW_PAINT_MSG = "PAINTBRUSH | L-Click: Paint | R-Click: Erase | M-Click: Pan | Scroll: Zoom | [P]: Exit Paint"

from src.batch_editing import BatchProcessingDialog

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
        self.crop_mode = False
        self.image_exif = None
        self.blur_radius = 30
        self.outline_color = QColor(0, 0, 0)
        self.inner_glow_color = QColor(255, 255, 255)
        self.tint_color = QColor(255, 200, 150)

        
        self.working_orig_image = None
        self.raw_original_image = None # raw unedited source image. needs better naming

        self.adjustment_source_np = None  # adjustment source as BGRA NumPy array for performance
         # Timer for debouncing adjustment updates
        self.adjust_timer = QTimer()
        self.adjust_timer.setSingleShot(True)
        self.adjust_timer.setInterval(10) # Low interval for near real-time feel
        self.adjust_timer.timeout.connect(self.apply_adjustments)

        # New timer specifically for the output view
        # output view image rendering can be slow, so it doesnt need to update as frequently
        self.output_refresh_timer = QTimer()
        self.output_refresh_timer.setSingleShot(True)
        self.output_refresh_timer.setInterval(300)
        self.output_refresh_timer.timeout.connect(self.update_output_preview)
        self.output_refresh_timer.timeout.connect(self.trigger_refinement_update)

        self.working_mask = None
        self.last_trimap = None

        self.model_output_mask = None
        try: pyi_splash.update_text("Loading UI")
        except: pass
        self.init_ui()

        if cli_args.binary:
            self.chk_binarise_mask.setChecked(True)
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

        self.refinement_timer = QTimer()
        self.refinement_timer.setSingleShot(True)
        self.refinement_timer.setInterval(500) # 500ms delay to prevent spamming ONNX models
        self.refinement_timer.timeout.connect(lambda: self.modify_mask(op="live_preview"))

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

            if cli_args.load_trimap:
                QTimer.singleShot(70, lambda: self.load_associated_trimap(self.image_paths[0]))
        
        saved_theme = self.settings.value("theme", "dark")
        self.set_theme(saved_theme)


        try: pyi_splash.close()
        except: pass



    def init_ui(self):
        self.setAcceptDrops(True)
        main = QWidget()
        self.setCentralWidget(main)
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Toolbar
        self.create_global_toolbar()

        # Sidebar
        self.sidebar_container = QWidget()
        self.sidebar_container.setMinimumWidth(300)
        sidebar_layout = QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)

        self.tabs = QTabWidget()

        self.tabs.addTab(self.create_adjust_tab(), "Adjust")
        self.tabs.addTab(self.create_ai_mask_tab(), "Mask Gen.")
        self.tabs.addTab(self.create_refine_tab(), "Refine")
        self.tabs.addTab(self.create_export_tab(), "Export")

        self.tabs.setCurrentIndex(1)

        sidebar_layout.addWidget(self.tabs)

        sidebar_layout.addWidget(self.create_commit_widget())

             
        # Input/Output Views

        # Custom splitter with orientation toggle on the divider
        self.in_out_splitter = OrientationSplitter(Qt.Orientation.Horizontal)
        self.in_out_splitter.setChildrenCollapsible(False)

        # Input view
        self.scene_input = QGraphicsScene()
        self.view_input = SynchronisedGraphicsView(self.scene_input, name="Input View") 
        self.view_input.set_controller(self)
        self.view_input.setAcceptDrops(True)

        self.input_pixmap_item = QGraphicsPixmapItem()
        self.scene_input.addItem(self.input_pixmap_item)
        
        self.overlay_pixmap_item = QGraphicsPixmapItem()
        self.overlay_pixmap_item.setOpacity(0.5)
        self.scene_input.addItem(self.overlay_pixmap_item)
        
        self.trimap_overlay_item = QGraphicsPixmapItem()
        self.trimap_overlay_item.setOpacity(0.7)
        self.scene_input.addItem(self.trimap_overlay_item)

        self.in_out_splitter.addWidget(self.view_input)

        # Input view UI additions
        self.chk_input_mask_only = QCheckBox("Show Only Model Output", self.view_input)
        self.chk_input_mask_only.toggled.connect(self.show_mask_overlay)
        self.chk_input_mask_only.move(10, 10)
        self.chk_input_mask_only.setStyleSheet("""
            QCheckBox {
                background-color: rgba(120, 120, 120, 120);
                color: white;
                padding: 4px;
                border-radius: 4px;
            }
        """)


        # Output View
        self.scene_output = QGraphicsScene()
        self.view_output = SynchronisedGraphicsView(self.scene_output, name="Output View")
        self.view_output.set_controller(self) 
        self.view_output.setAcceptDrops(True)
        

        self.chk_show_mask = QCheckBox("Show Global Mask", self.view_output)
        self.chk_show_mask.toggled.connect(self.update_output_preview)
        self.chk_show_mask.move(10, 10)
        self.chk_show_mask.setStyleSheet("""
            QCheckBox {
                background-color: rgba(120, 120, 120, 120);
                color: white;
                padding: 4px;
                border-radius: 4px;
            }
        """)

        self.chk_show_partial_alpha = QCheckBox("Show Transparency in Red (alpha = 1 to 244)", self.view_output)
        self.chk_show_partial_alpha.toggled.connect(self.update_output_preview)
        self.chk_show_partial_alpha.move(20, 40)
        self.chk_show_partial_alpha.setStyleSheet("""
            QCheckBox {
                background-color: rgba(120, 120, 120, 120);
                color: white;
                padding: 4px;
                border-radius: 4px;
            }
        """)

        self.chk_show_partial_alpha.setVisible(self.chk_show_mask.isChecked())
        self.chk_show_mask.toggled.connect(self.chk_show_partial_alpha.setVisible)
        
        # the output image
        self.output_pixmap_item = QGraphicsPixmapItem()
        self.scene_output.addItem(self.output_pixmap_item)

        # overlay when "trim transparent pixels" is selected
        self.output_crop_overlay = QGraphicsPathItem()
        self.output_crop_overlay.setBrush(QBrush(QColor(0, 0, 0, 160))) # Semi-transparent black
        self.output_crop_overlay.setPen(QPen(QColor(255, 255, 255, 200), 1, Qt.PenStyle.DashLine)) # Thin dashed white border
        self.output_crop_overlay.setZValue(1000) # Ensure it is above the image
        self.output_crop_overlay.hide()
        self.scene_output.addItem(self.output_crop_overlay)

        self.in_out_splitter.addWidget(self.view_output)

        self.in_out_splitter.toggle_button.clicked.connect(self.toggle_splitter_orientation)

        self.view_input.set_sibling(self.view_output)
        self.view_output.set_sibling(self.view_input)

        # status bar help messages on view mouseover
        self.view_input.viewport().installEventFilter(self)
        self.view_output.viewport().installEventFilter(self)

        # Holds the views (top) and the thumbnails (bottom)
        views_thumbs_widget = QWidget()
        views_thumbs_layout = QVBoxLayout(views_thumbs_widget)
        views_thumbs_layout.setContentsMargins(0, 0, 0, 0)
        views_thumbs_layout.setSpacing(2) # Can add small gap between views and thumbnails

        # Initialise the thumbnail strip
        self.thumbnail_strip = ThumbnailList()
        self.thumbnail_strip.itemClicked.connect(self.on_thumbnail_clicked)

        # Bottom Strip (Thumbs + Batch Button)
        bottom_strip_widget = QWidget()
        bottom_strip_layout = QHBoxLayout(bottom_strip_widget)
        bottom_strip_layout.setContentsMargins(0, 0, 0, 0)
        bottom_strip_layout.setSpacing(2)
        
        bottom_strip_layout.addWidget(self.thumbnail_strip, 1) # Expand
        
        self.btn_batch = QPushButton("Batch\nProcess")
        self.btn_batch.setToolTip("Run current settings on all loaded images")
        self.btn_batch.clicked.connect(self.open_batch_dialog)
        self.btn_batch.setFixedWidth(80) 
        self.btn_batch.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        # Style to look actionable but distinct
        self.btn_batch.setStyleSheet("background-color: #444; color: white; font-weight: bold; border: 1px solid #555;")
        
        bottom_strip_layout.addWidget(self.btn_batch)

        # Add the splitter to the top, thumbnails to the bottom
        views_thumbs_layout.addWidget(self.in_out_splitter, 1) # '1' ensures views take most space
        views_thumbs_layout.addWidget(bottom_strip_widget)

        # Assemble sidebar + views using a splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.addWidget(self.sidebar_container)
        self.main_splitter.addWidget(views_thumbs_widget)
        
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([350, 1250])
        
        main_layout.addWidget(self.main_splitter)

        # status bar 
        self.status_label = QLabel("Ready")
        self.zoom_label = QLabel("Zoom: 100%")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Sets "Indeterminate" (bouncing) mode
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.hide()

        self.statusBar().addWidget(self.status_label)
        self.statusBar().addPermanentWidget(self.progress_bar) # Add to right side
        self.statusBar().addPermanentWidget(self.zoom_label)

        self.view_input.set_placeholder(
            "<b>Input</b><br><br>"
            "Generated masks from AI models will be displayed here.<br>"
            "Models are run on the viewport. Zoom in/pan the image to focus<br>"
            "on your subject or work on smaller areas in detail."
        )
        self.view_output.set_placeholder(
            "<b>Output</b><br><br>"
            "Use generated masks to build up an output image."
        )
        
        self.toggle_splitter_orientation(initial_setup=True)
    
    def create_global_toolbar(self):
        toolbar = QToolBar("Session Control")
        toolbar.setMovable(False)
        toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)

        open_img_act = QAction("Open Image(s)", self)
        open_img_act.setToolTip("Open Image(s)")
        open_img_act.triggered.connect(self.load_image_dialog)
        toolbar.addAction(open_img_act)
        open_clip_act = QAction("Open Clipboard", self)
        open_clip_act.setToolTip("Open Clipboard")
        open_clip_act.triggered.connect(self.load_clipboard)
        toolbar.addAction(open_clip_act)
        # toolbar.addSeparator()
        # edit_image_act = QAction("Edit Image (Curves/WB)", self)
        # edit_image_act.triggered.connect(self.open_image_editor)
        # toolbar.addAction(edit_image_act)
        
        toolbar.addSeparator()

        load_mask_act = QAction("Import Saved Mask", self)
        load_mask_act.setToolTip("Import an existing .png mask into the global working composite")
        toolbar.addAction(load_mask_act)
        load_mask_act.triggered.connect(self.load_mask_dialog)

        toolbar.addSeparator()
        undo_act = QAction("↶", self)
        undo_act.setToolTip("Undo")
        undo_act.triggered.connect(self.undo)
        toolbar.addAction(undo_act)

        redo_act = QAction("↷", self)
        redo_act.setToolTip("Redo")
        redo_act.triggered.connect(self.redo)
        toolbar.addAction(redo_act)

        toolbar.addSeparator()

        # Global Toggle
        self.chk_paint = QCheckBox("Paintbrush/Smart Refine (P)")
        self.chk_paint.setToolTip("Manually draw to touch-up areas. Hold CTRL while painting for Smart Refinement of that area")

        toolbar.addWidget(self.chk_paint)
        self.chk_paint.toggled.connect(self.toggle_paint_mode)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        
        toolbar.addSeparator()

        toolbar.addAction("Settings ⚙️").triggered.connect(self.open_settings)

        toolbar.addAction("Help/About").triggered.connect(self.show_help)

    def open_batch_dialog(self):
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please open images first.")
            return

        gen_settings = self.get_generation_settings()
        render_settings = self.get_render_settings()
        adj_settings = self.get_adjustment_params()
        export_settings = self.get_export_settings()
        
        dlg = BatchProcessingDialog(self, self.image_paths, self.model_manager,
                                    gen_settings, render_settings, adj_settings, export_settings)
        dlg.exec()

    def get_generation_settings(self):
        """
        Captures settings related to mask generation (inference + refinement).
        """
        matting_enabled = self.chk_alpha_matting.isChecked()
        return {
            "model_name": self.combo_whole.currentText(),
            "provider_data": self.combo_auto_model_EP.currentData(),
            "use_2step": self.chk_2step_auto.isChecked(),
            "matting": {
                "enabled": matting_enabled,
                "algorithm": self.combo_matting_algorithm.currentText(),
                # Assuming simple auto trimap for batch
                "fg_erode": self.sl_fg_erode.value(),
                "bg_erode": self.sl_bg_erode.value(),
                "provider_data": self.combo_auto_model_EP.currentData(),
                "longest_edge_limit": int(self.settings.value("matting_longest_edge", 1024))
            },
            "soften": self.chk_soften.isChecked(),
            "soften_radius": 1.5, # Could be made configurable
            "binarise": self.chk_binarise_mask.isChecked()
        }
    
    def get_render_settings(self):
        """
        Captures all current UI settings related to image composition into a dictionary.
        """
        return {
            "clean_alpha": self.chk_clean_alpha.isChecked(),
            "foreground_correction": {
                "enabled": self.chk_estimate_foreground.isChecked(),
                "algorithm": self.settings.value("fg_correction_algo", "ml")
            },
            "tint": {
                "enabled": self.chk_tint.isChecked(),
                "color": (self.tint_color.red(), self.tint_color.green(), self.tint_color.blue()),
                "amount": self.sl_tint_amt.value() / 100.0
            },
            "outline": {
                "enabled": self.chk_outline.isChecked(),
                "size": self.sl_outline_size.value(),
                "color": (self.outline_color.red(), self.outline_color.green(), self.outline_color.blue()),
                "threshold": self.sl_outline_thresh.value(),
                "opacity": self.sl_outline_op.value()
            },
            "shadow": {
                "enabled": self.chk_shadow.isChecked(),
                "opacity": self.sl_s_op.value(),
                "radius": self.sl_s_r.value(),
                "x": self.sl_s_x.value(),
                "y": self.sl_s_y.value()
            },
            "background": {
                "type": self.combo_bg_color.currentText(),
                "blur_radius": getattr(self, 'blur_radius', 30),
                "color": self.combo_bg_color.currentText()
            },
            "inner_glow": {
                "enabled": self.chk_inner_glow.isChecked(),
                "size": self.sl_ig_size.value(),
                "color": (self.inner_glow_color.red(), self.inner_glow_color.green(), self.inner_glow_color.blue()),
                "threshold": self.sl_ig_thresh.value(),
                "opacity": self.sl_ig_op.value()
            }
        }

    def get_export_settings(self):
        return {
            "format": self.combo_export_fmt.currentData(),
            "quality": self.sl_export_quality.value(),
            "save_mask": self.chk_export_mask.isChecked(),
            "trim": self.chk_export_trim.isChecked()
        }

    def create_adjust_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(0) 

        lbl_desc = QLabel("Colour and tone adjustments applied to the image before mask generation.")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        layout.addSpacing(10)


        self.adj_sliders = {}
        self.adj_slider_params = {
            # (Min, Max, Default)
            'highlight': (10, 200, 100), 
            'midtone': (10, 200, 100), 
            'shadow': (10, 300, 100),
            'tone_curve': (1, 50, 10),     
            'brightness': (10, 200, 100), 
            'contrast': (10, 200, 100),
            'saturation': (0, 200, 100),
            'white_balance': (2000, 10000, 6500), 
            'unsharp_radius': (1, 100, 1), 
            'unsharp_amount': (0, 500, 0), 
            'unsharp_threshold': (0, 255, 0)
        }

        for param, (min_v, max_v, default) in self.adj_slider_params.items():
            lbl = QLabel(param.replace("_", " ").capitalize())
            layout.addWidget(lbl)

            h_row_layout = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default)
            
            val_display = QLabel(str(default))
            val_display.setFixedWidth(40)
            val_display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            slider.valueChanged.connect(lambda v, l=val_display: l.setText(str(v)))
            slider.valueChanged.connect(lambda _: self.adjust_timer.start())
            
            h_row_layout.addWidget(slider)
            h_row_layout.addWidget(val_display)
            layout.addLayout(h_row_layout)
            self.adj_sliders[param] = slider

        layout.addSpacing(10)
        btn_reset_adj = QPushButton("Reset Adjustments")
        btn_reset_adj.clicked.connect(self.reset_adjustment_sliders)
        layout.addWidget(btn_reset_adj)

        layout.addSpacing(10)

        rot_layout = QHBoxLayout()
        btn_rot_l = QPushButton("Rotate Left ↶")
        btn_rot_l.setToolTip("This will reset Undo history")
        btn_rot_l.clicked.connect(lambda: self.rotate_image(direction="left"))
        
        btn_rot_r = QPushButton("Rotate Right ↷")
        btn_rot_r.setToolTip("This will reset Undo history")
        btn_rot_r.clicked.connect(lambda: self.rotate_image(direction="right"))
        
        rot_layout.addWidget(btn_rot_l)
        rot_layout.addWidget(btn_rot_r)
        layout.addLayout(rot_layout)

        layout.addSpacing(10)

        lbl_crop_warning = QLabel("<i>Inpainting or Cropping are destructive actions and cannot be reverted without re-opening the image.</i>")
        lbl_crop_warning.setWordWrap(True)
        lbl_crop_warning.setStyleSheet("color: #888; margin-bottom: 5px;")
        layout.addWidget(lbl_crop_warning)
        
        # Inpainting Section
        self.btn_inpaint = QPushButton("Inpaint / Remove Object")
        self.btn_inpaint.setToolTip("Opens the Inpainting dialog. Allows you to draw over an object to remove it.\n"
                                    "Requires the LaMa model (Download via Settings).")
        self.btn_inpaint.clicked.connect(self.open_inpainting_dialog)
        layout.addWidget(self.btn_inpaint)

        layout.addSpacing(10)

        crop_layout = QHBoxLayout()
        
        self.btn_toggle_crop = QPushButton("Crop Image ✂")
        self.btn_toggle_crop.setCheckable(True)
        self.btn_toggle_crop.toggled.connect(self.toggle_crop_mode)
        self.btn_toggle_crop.setToolTip("Draw a box on the input view to crop the original image.\n"
                                        "Note: This is a destructive action and will clear undo history.")
        
        self.btn_apply_crop = QPushButton("Apply Crop")
        self.btn_apply_crop.clicked.connect(self.apply_crop)
        self.btn_apply_crop.setEnabled(False) # Disabled until mode is active
        self.btn_apply_crop.setStyleSheet("background-color: #2e7d32; color: white;") # Green to signify action
        self.btn_apply_crop.hide()

        crop_layout.addWidget(self.btn_toggle_crop)
        crop_layout.addWidget(self.btn_apply_crop)
        
        layout.addLayout(crop_layout)

        

        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def reset_adjustment_sliders(self):
        self.adjust_timer.stop()
        self.output_refresh_timer.stop() 

        for param, (_, _, default) in self.adj_slider_params.items():
            self.adj_sliders[param].blockSignals(True)
            self.adj_sliders[param].setValue(default)
            self.adj_sliders[param].blockSignals(False)
        
        # Update the labels and the image
        for slider in self.adj_sliders.values():
            slider.valueChanged.emit(slider.value())
        
        self.apply_adjustments()

    def get_adjustment_params(self):
        return {param: slider.value() for param, slider in self.adj_sliders.items()}

    def apply_adjustments(self):
        if self.adjustment_source_np is None:
            return

        params = self.get_adjustment_params()
        
        processed_np = apply_tone_sharpness(self.adjustment_source_np, params)

        # Clear sam encoder, since the results depend on the src image
        self.model_manager.clear_sam_cache(clear_loaded_models=False)
        
        # inefficient, but until I remove all legacy PIL image operations, it is necessary
        self.working_orig_image = Image.fromarray(cv2.cvtColor(processed_np, cv2.COLOR_BGRA2RGBA))
        
        self.update_input_view(reset_zoom=False)
        
        # Delay the output view because some operations in render_output_image are slow
        self.output_refresh_timer.start()

    def toggle_crop_mode(self, checked):
        self.crop_mode = checked
        
        if checked:
            self.view_input.setCursor(Qt.CursorShape.CrossCursor)
            self.btn_toggle_crop.setText("Cancel Crop")
            self.status_label.setText("CROP MODE | Drag to draw crop area | Click 'Apply Crop' to finish")
            self.btn_apply_crop.show()
            self.btn_apply_crop.setEnabled(True)
            self.chk_paint.setEnabled(False)
            if self.chk_paint.isChecked():
                self.chk_paint.setChecked(False)
            
            # Disable SAM interactions visually
            self.combo_sam.setEnabled(False)
            
        else:
            self.btn_toggle_crop.setText("Crop Image ✂")
            self.view_input.setCursor(Qt.CursorShape.ArrowCursor)
            self.status_label.setText("Ready")
            self.btn_apply_crop.hide()
            self.btn_apply_crop.setEnabled(False)
            self.chk_paint.setEnabled(True)
            self.combo_sam.setEnabled(True)
            
            # Hide the rect in the view
            if hasattr(self.view_input, 'crop_rect_item'):
                self.view_input.crop_rect_item.hide()

    def apply_crop(self):
        if not self.crop_mode: return
        
        # Get the rect from the view item
        rect_item = self.view_input.crop_rect_item
        if not rect_item.isVisible() or rect_item.rect().isEmpty():
            QMessageBox.information(self, "No Selection", "Please drag a box on the image to select the crop area.")
            return

        rect = rect_item.rect().normalized()
        
        # Map scene coordinates to image pixels
        x, y, w, h = int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        
        # Bounds check
        if w <= 0 or h <= 0: return
        
        # Clamp to image size
        img_w, img_h = self.raw_original_image.size
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        if w <= 0 or h <= 0: return

        # Perform the Crop
        box = (x, y, x+w, y+h)
        
        # Crop the Master (the raw, unadjusted image)
        self.working_orig_image = self.raw_original_image.crop(box)
        
        # Update the NumPy buffer from the cropped raw image, so no adjustments are baked in
        self.adjustment_source_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.working_orig_image), cv2.COLOR_RGBA2BGRA)
        )
        
        if self.working_mask:
            self.working_mask = self.working_mask.crop(box)
            
        if self.model_output_mask:
            self.model_output_mask = self.model_output_mask.crop(box)

        if hasattr(self, 'user_trimap') and self.user_trimap:
            self.user_trimap = self.user_trimap.crop(box)
            self.last_trimap = np.array(self.user_trimap)

        # Since dimensions changed, old undo history is invalid
        # Possibly could be implemented properly
        self.undo_history = [self.working_mask.copy()] 
        self.redo_history = []
        
        # Clear SAM points as they are now misaligned
        self.coordinates = []
        self.labels = []

        # Re-initialise the paint scratchpad
        new_size = self.working_orig_image.size
        self.paint_image = QImage(new_size[0], new_size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)
        
        # Generate the new self.original_image
        self.apply_adjustments()
        
        # Reset UI
        self.btn_toggle_crop.setChecked(False) # This triggers toggle_crop_mode(False)
        self.update_output_preview()
        self.clear_overlay() # Clears points from screen

        # Ensure view bounds are updated for the new size
        self.view_input.setSceneRect(self.input_pixmap_item.boundingRect())
        self.view_input.fitInView(self.input_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        
        self.status_label.setText(f"Image Cropped to {w}x{h}")

    def rotate_image(self, direction):
        if not self.raw_original_image:
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if direction == "left":
                pil_transpose = Image.Transpose.ROTATE_90
                cv2_code = cv2.ROTATE_90_COUNTERCLOCKWISE
            else: # right
                pil_transpose = Image.Transpose.ROTATE_270
                cv2_code = cv2.ROTATE_90_CLOCKWISE

            self.raw_original_image = self.raw_original_image.transpose(pil_transpose)

            self.adjustment_source_np = cv2.rotate(self.adjustment_source_np, cv2_code)

            if self.working_mask:
                self.working_mask = self.working_mask.transpose(pil_transpose)
            if self.model_output_mask:
                self.model_output_mask = self.model_output_mask.transpose(pil_transpose)
            if hasattr(self, 'user_trimap') and self.user_trimap:
                self.user_trimap = self.user_trimap.transpose(pil_transpose)

            # Re-initialise paint scratchpad for new orientation
            new_size = self.raw_original_image.size
            self.paint_image = QImage(new_size[0], new_size[1], QImage.Format.Format_ARGB32_Premultiplied)
            self.paint_image.fill(Qt.GlobalColor.transparent)

            # Clear state that is no longer valid for the new orientation
            # undo could probably be implemented
            self.undo_history = [self.working_mask.copy()]
            self.redo_history = []
            self.coordinates = []
            self.labels = []
            self.clear_overlay() # Remove SAM dots from view

            # Re-apply adjustments to update self.original_image
            self.apply_adjustments()
            
            self.view_input.fitInView(self.input_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.update_output_preview()

            self.status_label.setText(f"Rotated image {direction}")

        finally:
            QApplication.restoreOverrideCursor()

    def open_inpainting_dialog(self):
        if not self.working_orig_image:
            return
            
        # Check for models
        root = self.model_manager.model_root_dir
        lama_exists = (self.model_manager.check_is_cached("lama", self.settings.value("exec_short_code", "cpu")) or 
                       os.path.exists(os.path.join(root, "lama.onnx")))
        
        deepfill_exists = False
        df_variants = [
            'deepfillv2_celeba_256x256',
            'deepfillv2_places_256x256', 
            'deepfillv2_places_512x512', 
            'deepfillv2_places_1024x1024'
        ]
        for variant in df_variants:
            if os.path.exists(os.path.join(root, f"{variant}.onnx")):
                deepfill_exists = True
                break
        
        if not lama_exists and not deepfill_exists:
             QMessageBox.information(self, "Model Missing", "Please download an Inpainting model (LaMa or DeepFill) from the Settings -> Inpainting Models tab first.")
             return

        # Use the Automatic provider for inpainting
        provider_data = self.combo_auto_model_EP.currentData()
        
        # Apply current adjustments to get the image as seen on screen
        # Unsure if should pass actual original image, or what is on screen, but doing this for now
        if self.adjustment_source_np is not None:
             params = self.get_adjustment_params()
             processed_np = apply_tone_sharpness(self.adjustment_source_np, params)
             img_to_edit = Image.fromarray(cv2.cvtColor(processed_np, cv2.COLOR_BGRA2RGBA))
        else:
             img_to_edit = self.working_orig_image.copy()

        dlg = InpaintingDialog(img_to_edit, self.model_manager, provider_data, self)
        
        if dlg.exec():
            # User clicked Apply
            new_image = dlg.get_result()
            
            # This is a destructive action on the "Original" image
            self.working_orig_image = new_image
            self.raw_original_image = new_image
            
            self.adjustment_source_np = np.ascontiguousarray(
                cv2.cvtColor(np.array(self.working_orig_image), cv2.COLOR_RGBA2BGRA)
            )
            
            # Since we appled adjustments, reset here
            # If change to send actual original image, don't reset
            self.reset_adjustment_sliders()
            
            self.update_output_preview()
            
            self.status_label.setText("Inpainting applied. Image updated.")

    def create_ai_mask_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)


        # Mask generation model selection

        h_models_header = QHBoxLayout()
        lbl_models = QLabel("")
        lbl_models.setContentsMargins(3, 0, 0, 0)
        h_models_header.addWidget(lbl_models)
        h_models_header.addStretch()

        self.btn_download = QPushButton("Download AI Models 📥")
        self.btn_download.setToolTip("Download Models...")
        self.btn_download.clicked.connect(self.open_settings)
        layout.addWidget(self.btn_download)
        
        layout.addSpacing(20)

        
        lbl_sam = QLabel("<b>INTERACTIVE (SAM)<b>")
        lbl_sam.setToolTip("<b>Segment Anything Models</b><br>"
                           "These require you to interact with the image.<br>"
                           "<i>Usage: Left-click to add points, right-click to add negative (avoid) points, or drag to draw boxes around the subject.</i><br><br>"
                           "Disc drive icons show models that have saved optimised versions cached.")
        layout.addWidget(lbl_sam)
        lbl_sam_desc = QLabel("Point and click models that let you choose parts of the image to add/subtract")
        lbl_sam_desc.setWordWrap(True)
        layout.addWidget(lbl_sam_desc)

        self.combo_sam = QComboBox()
        self.combo_sam.setToolTip(lbl_sam.toolTip())
        self.populate_sam_models()
        self.combo_sam.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_sam.setMinimumContentsLength(1) 
        layout.addWidget(self.combo_sam)

        # Add vertical space after the SAM elements
        layout.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))


        lbl_auto = QLabel("<b>AUTOMATIC<b>")
        lbl_auto.setToolTip("<b>Automatic Models</b><br>"
                             "These run automatically on the image/current zoomed view.<br>"
                             "<i>Usage: Select a model and click 'Run Automatic'. No points needed.</i><br><br>"
                             "Disc drive icons show models that have saved optimised versions cached.")
        layout.addWidget(lbl_auto)
        lbl_auto_desc = QLabel("Models that perform their best guess bg removal for the image.")
        lbl_auto_desc.setWordWrap(True)
        layout.addWidget(lbl_auto_desc)

        # Whole Image Combo
        self.combo_whole = QComboBox()
        self.combo_whole.setToolTip(lbl_auto.toolTip()) # Reuse the tooltip
        self.populate_whole_models()
        self.combo_whole.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_whole.setMinimumContentsLength(1)
        layout.addWidget(self.combo_whole)
        
        # Run Model Button and layout adjustment
        h_whole_model = QHBoxLayout()
        self.btn_whole = QPushButton("Run Model (M)"); self.btn_whole.clicked.connect(lambda: self.run_automatic_model())
        h_whole_model.addWidget(self.combo_whole)
        h_whole_model.addWidget(self.btn_whole)
        layout.addLayout(h_whole_model)

        self.chk_2step_auto = QCheckBox("2-Step (Find -> Re-run)")
        self.chk_2step_auto.setToolTip("Runs the model twice: first on the entire image to find the subject,\n "
                                       "then a second time on a zoomed crop of the subject for maximum detail."
                                       "\n\nNOTE: This is different to typical usage, where the model is run on what you are currently zoomed in to,\n"
                                       "which may instead be the optimal way to get a higher quality mask")
        layout.addWidget(self.chk_2step_auto)

        layout.addSpacing(20)


        layout.addSpacing(20)


        btn_clr = QPushButton("Clear SAM Clicks/ Model Masks (C)"); btn_clr.clicked.connect(self.clear_overlay)
        layout.addWidget(btn_clr)

        layout.addSpacing(20)

        # Hardware Acceleration
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

        # End Hardware Acceleration
        layout.addWidget(self.hw_options_frame)
        
        # Persistence for Collapsible Frame
        hw_collapsed = self.settings.value("hw_options_collapsed", True, type=bool)
        self.hw_options_frame.set_collapsed(hw_collapsed)
        self.hw_options_frame.toggled.connect(
            lambda collapsed: self.settings.setValue("hw_options_collapsed", collapsed)
        )

        last_auto_cache_mode = self.settings.value("auto_ram_cache_mode", 1, type=int)
        self.auto_cache_group.blockSignals(True)
        self.auto_cache_group.button(last_auto_cache_mode).setChecked(True)
        self.auto_cache_group.blockSignals(False)

        self.model_manager.auto_cache_mode = last_auto_cache_mode

        self.trt_cache_option_visibility() # Initial check for TensorRT

        # End Hardware Acceleration


        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def create_refine_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        

        layout.addSpacing(10)

        

        lbl_global_modifiers = QLabel("<b>GLOBAL REFINEMENTS</b>")
        lbl_global_modifiers.setToolTip("Applied to the entire image")
        lbl_global_modifiers.setContentsMargins(3, 0, 0, 0)
        layout.addWidget(lbl_global_modifiers)

        self.chk_clean_alpha = QCheckBox("Clean Transparency")
        self.chk_clean_alpha.setToolTip("Remove nearly transparent pixels (alpha < 5) and solidify nearly opaque (alpha >250) to fully opaque (255).\n"
                                        "You can view the partial transparency via View Global Mask on the output canvas\n"
                                        "Recommended to leave on.")
        self.chk_clean_alpha.setChecked(self.settings.value("clean_alpha", True, type=bool))
        # In your UI setup code where you define the checkbox:
        self.chk_clean_alpha.toggled.connect(
            lambda checked: self.settings.setValue("clean_alpha", checked)
        )
        self.chk_clean_alpha.toggled.connect(self.update_output_preview)
        layout.addWidget(self.chk_clean_alpha)

        self.chk_estimate_foreground = QCheckBox("Foreground Edge \nColour Correction (Slow)")
        self.chk_estimate_foreground.setToolTip("Recalculates edge colors to remove halos or fringes from the original background.\n"
                                                "Recommended for soft edges such as hair")
        layout.addWidget(self.chk_estimate_foreground)
        self.chk_estimate_foreground.toggled.connect(self.update_output_preview)

        layout.addSpacing(10)

        lbl_modifiers = QLabel("<b>MODEL OUTPUT MASK REFINEMENTS</b>")
        lbl_modifiers.setContentsMargins(3, 0, 0, 0)
        layout.addWidget(lbl_modifiers)

        lbl_desc = QLabel("Applied to the AI model's mask when adding or subtracting to the output image.")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        #layout.addSpacing(10)

        # indent smart refine to highlight (crudely) that they are part of the live preview
        indent_container = QWidget()
        indent_layout = QVBoxLayout(indent_container)
        indent_layout.setContentsMargins(10, 0, 0, 0)



        self.chk_binarise_mask = QCheckBox("Remove Mask Partial Transparency")
        self.chk_binarise_mask.toggled.connect(self.trigger_refinement_update)
        indent_layout.addWidget(self.chk_binarise_mask)

        self.chk_soften = QCheckBox("Soften Mask/Paintbrush Edges")
        soften_checked = self.settings.value("soften_mask", False, type=bool)
        self.chk_soften.setChecked(soften_checked)
        self.chk_soften.toggled.connect(lambda checked: self.settings.setValue("soften_mask", checked))
        self.chk_soften.toggled.connect(self.trigger_refinement_update) 
        indent_layout.addWidget(self.chk_soften)
        
        indent_layout.addSpacing(20)


        matt_tt = "Uses a matting algorithm to estimate the transparency of mask edges.\n" + "This can improve the quality of detailed edges such as hair, especially when using binary mask models like SAM.\n" + "This requires a trimap (a foreground, unknown, background mask), either estimated from a SAM or automatic models, or manually drawn.\n" + "Alpha matting is computationally expensive and is only applied when 'Add' or 'Subtract' is clicked. Undo if the effect is unsatisfactory"
        lbl_alpha = QLabel("<b>SMART REFINE (Alpha Matting)</b>")
        lbl_alpha.setToolTip(matt_tt)
        indent_layout.addWidget(lbl_alpha)
        lbl_alpha_desc = QLabel("Uses specialised algorithms to refine difficult areas like hair and fur. ViTMatte is recommended")
        lbl_alpha_desc.setWordWrap(True)
        lbl_alpha_desc.setToolTip(matt_tt)
        indent_layout.addWidget(lbl_alpha_desc)

        btn_refine_download = QPushButton("Download Refinement Models 📥")
        btn_refine_download.setToolTip("Download Refinement Models. VitMatte Small is recommended")
        btn_refine_download.clicked.connect(self.open_settings)
        indent_layout.addWidget(btn_refine_download)


        ma_layout = QHBoxLayout()

        matting_label = QLabel("Algorithm:")
        matting_tt = "Additional models can be downloaded using the model manager.\nThe default included PyMatting algo can be very slow on large images.\nViTMatte (model downloader) can be much faster and far more accurate"
        matting_label.setToolTip(matting_tt)
        ma_layout.addWidget(matting_label, 0)


        self.combo_matting_algorithm = QComboBox()
        self.combo_matting_algorithm.setToolTip(matting_tt)
        self.populate_matting_models()
        self.combo_matting_algorithm.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_matting_algorithm.setMinimumContentsLength(1)
        self.combo_matting_algorithm.currentIndexChanged.connect(self.trigger_refinement_update)
        ma_layout.addWidget(self.combo_matting_algorithm, 1)

        indent_layout.addLayout(ma_layout)

        self.chk_alpha_matting = QCheckBox("Enable Alpha Matting")
        self.chk_alpha_matting.setToolTip(matt_tt)
        self.chk_alpha_matting.toggled.connect(self.handle_alpha_matting_toggle)
        self.chk_alpha_matting.toggled.connect(self.trigger_refinement_update) 
        indent_layout.addWidget(self.chk_alpha_matting)


        # --- Alpha Matting Options Frame ---
        self.alpha_matting_frame = QFrame()
        am_layout = QVBoxLayout(self.alpha_matting_frame)
        am_layout.setContentsMargins(15, 5, 0, 5) # Indent options slightly

        self.chk_show_trimap = QCheckBox("Show Trimap on Input")
        self.chk_show_trimap.setToolTip("Displays the generated trimap on the input view.\n"
                                        "White = Foreground, Blue = Unknown (semi transparent, e.g. hair edges), Black = Background")
        self.chk_show_trimap.toggled.connect(self.toggle_trimap_display)
        am_layout.addWidget(self.chk_show_trimap)

        # --- Trimap Source Radio Buttons ---
        lbl_tri_src = QLabel("<b>Trimap Source:</b>")
        lbl_tri_src.setToolTip("A trimap is a guidance mask that specifies what is definite foreground, definite background, and 'unknown/mixed' for the model to calculate")
        am_layout.addWidget(lbl_tri_src)
        self.trimap_mode_group = QButtonGroup(self)
        self.rb_trimap_auto = QRadioButton("Automatic \n (model output border expand)")
        self.rb_trimap_custom = QRadioButton("Custom (user-drawn)")
        
        self.trimap_mode_group.addButton(self.rb_trimap_auto)
        self.trimap_mode_group.addButton(self.rb_trimap_custom)
        
        self.rb_trimap_auto.setChecked(True)

        am_layout.addWidget(self.rb_trimap_auto)
        
        self.trimap_mode_group.buttonToggled.connect(self.on_trimap_mode_changed)
        self.trimap_mode_group.buttonToggled.connect(self.trigger_refinement_update)
        
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
            slider.valueChanged.connect(lambda val, l=label, txt=lbl_text: (l.setText(f"{txt}: {val}"), 
                                                                            self.update_trimap_preview_throttled(),
                                                                            self.trigger_refinement_update()))
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

        am_layout.addWidget(self.rb_trimap_custom)
        
        self.btn_edit_trimap = QPushButton("Open Trimap Editor...")
        self.btn_edit_trimap.clicked.connect(self.open_trimap_editor)
        self.btn_edit_trimap.setEnabled(False)
        am_layout.addWidget(self.btn_edit_trimap)
        
        


        indent_layout.addWidget(self.alpha_matting_frame)
        self.alpha_matting_frame.hide()

        # end alpha matting

        def make_brush_refine_slider(label_text, min_v, max_v, def_v, callback=None):
            h_layout = QHBoxLayout()
            label = QLabel(f"{label_text}: {def_v}")
            label.setMinimumWidth(120)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(def_v)
            
            def update_val(val):
                label.setText(f"{label_text}: {val}")
                if callback:
                    callback()

            slider.valueChanged.connect(update_val)
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            return label, slider, h_layout
        
        indent_layout.addSpacing(10)

        lbl_brush_refine = QLabel("<b>Smart Refine Brush (Experimental)</b>")
        indent_layout.addWidget(lbl_brush_refine)
        
        lbl_brush_hint = QLabel("Hold <b>Ctrl + Paint</b> to locally refine edges such as hair on the <i>output</i> image.")
        lbl_brush_hint.setWordWrap(True)
        #lbl_brush_hint.setStyleSheet("color: #2a82da;")
        indent_layout.addWidget(lbl_brush_hint)

        self.lbl_smart_padding, self.sl_smart_padding, pad_l = make_brush_refine_slider(
            "Context Padding", 1, 5, 3)
        self.sl_smart_padding.setToolTip("Determines how much 'Definite' background and foreground the AI sees around your brush stroke.\nCan have small impact on output quality")
        indent_layout.addLayout(pad_l)

        layout.addWidget(indent_container)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def handle_live_preview_toggle(self, checked):
        """Handles the live preview checkbox toggle."""
        if checked:
            if self.model_output_mask:
                self.modify_mask(op="live_preview")
        else:
            # Clear refined preview and reset UI opacity
            self.refined_preview_mask = None
            self.input_pixmap_item.show()
            self.input_pixmap_item.setOpacity(1.0)
            self.show_mask_overlay()
    
    def trigger_refinement_update(self):
        """Debounced trigger for updating the live preview mask."""
        if self.chk_live_preview.isChecked() and self.model_output_mask and not self.is_busy():
            self.refinement_timer.start()

    def create_export_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)


        lbl_options = QLabel("<b>OUTPUT STYLING</b>")
        layout.addWidget(lbl_options)

        bg_layout = QHBoxLayout()
        bg_label = QLabel("Background:")
        bg_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        bg_layout.addWidget(bg_label)
        self.combo_bg_color = QComboBox()
        self.combo_bg_color.addItems(["Transparent", "White", "Black", "Red", "Blue", 
                  "Orange", "Yellow", "Green", "Grey", 
                  "Lightgrey", "Brown", "Blurred (Slow)", "Original Image"])
        self.combo_bg_color.currentTextChanged.connect(self.handle_bg_change)
        bg_layout.addWidget(self.combo_bg_color)
        layout.addLayout(bg_layout)


        self.chk_shadow = QCheckBox("Drop Shadow")
        self.chk_shadow.toggled.connect(self.toggle_shadow_options)
        layout.addWidget(self.chk_shadow)

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
        self.lbl_s_r, self.sl_s_r, h_r_layout = make_slider_row("Blur Rad", 1, 200, 10)
        
        sf_layout.addLayout(h_op_layout)
        sf_layout.addLayout(h_x_layout)
        sf_layout.addLayout(h_y_layout)
        sf_layout.addLayout(h_r_layout)
            
        layout.addWidget(self.shadow_frame)
        self.shadow_frame.hide()

        #layout.addSpacing(10)

        # Outline around mask
        self.chk_outline = QCheckBox("Outline Cutout")
        self.chk_outline.toggled.connect(self.toggle_outline_options)
        layout.addWidget(self.chk_outline)

        self.outline_frame = QFrame()
        of_layout = QVBoxLayout(self.outline_frame)
        of_layout.setContentsMargins(0, 0, 0, 0)
        
        # Outline Size Slider
        self.lbl_outline_size, self.sl_outline_size, h_os_layout = make_slider_row("Size", 1, 100, 5)
        self.sl_outline_size.valueChanged.connect(lambda: self.shadow_timer.start())
        of_layout.addLayout(h_os_layout)

        # Outline Threshold Slider
        self.lbl_outline_thresh, self.sl_outline_thresh, h_ot_layout = make_slider_row("Threshold", 1, 254, 128)
        self.sl_outline_thresh.setToolTip("Lower values include more of the semi-transparent edges in the outline base.")
        self.sl_outline_thresh.valueChanged.connect(lambda: self.shadow_timer.start())
        of_layout.addLayout(h_ot_layout)

        # Outline Opacity Slider
        self.lbl_outline_op, self.sl_outline_op, h_oop_layout = make_slider_row("Opacity", 0, 255, 255)
        self.sl_outline_op.valueChanged.connect(lambda: self.shadow_timer.start())
        of_layout.addLayout(h_oop_layout)

        # Outline Colour Button
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Colour:"))
        self.btn_outline_color = QPushButton()
        self.btn_outline_color.setToolTip("Select the outline colour")
        self.btn_outline_color.clicked.connect(self.pick_outline_color)
        self.update_outline_color_button() 
        color_layout.addWidget(self.btn_outline_color)
        color_layout.addStretch()
        of_layout.addLayout(color_layout)

        layout.addWidget(self.outline_frame)
        self.outline_frame.hide()

        #layout.addStretch()

        # Inner Glow
        self.chk_inner_glow = QCheckBox("Inner Glow (Edge Light)")
        self.chk_inner_glow.toggled.connect(self.toggle_inner_glow_options)
        layout.addWidget(self.chk_inner_glow)

        self.inner_glow_frame = QFrame()
        ig_layout = QVBoxLayout(self.inner_glow_frame)
        ig_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_ig_size, self.sl_ig_size, h_ig_size = make_slider_row("Size", 1, 100, 10)
        self.sl_ig_size.valueChanged.connect(lambda: self.shadow_timer.start())
        ig_layout.addLayout(h_ig_size)

        self.lbl_ig_op, self.sl_ig_op, h_ig_op = make_slider_row("Opacity", 0, 255, 150)
        self.sl_ig_op.valueChanged.connect(lambda: self.shadow_timer.start())
        ig_layout.addLayout(h_ig_op)

        # New: Inner Glow Threshold Slider
        self.lbl_ig_thresh, self.sl_ig_thresh, h_ig_thresh = make_slider_row("Threshold", 1, 254, 128)
        self.sl_ig_thresh.setToolTip("Controls how far into the semi-transparency the glow starts.")
        self.sl_ig_thresh.valueChanged.connect(lambda: self.shadow_timer.start())
        ig_layout.addLayout(h_ig_thresh)

        # Color Picker for Inner Glow
        ig_col_layout = QHBoxLayout()
        ig_col_layout.addWidget(QLabel("Colour:"))
        self.btn_ig_color = QPushButton()
        self.btn_ig_color.clicked.connect(self.pick_inner_glow_color)
        self.update_inner_glow_color_button()
        ig_col_layout.addWidget(self.btn_ig_color)
        ig_col_layout.addStretch()
        ig_layout.addLayout(ig_col_layout)

        layout.addWidget(self.inner_glow_frame)
        self.inner_glow_frame.hide()

        # --- Subject Tint ---
        self.chk_tint = QCheckBox("Subject Colour Tint")
        self.chk_tint.toggled.connect(self.toggle_tint_options)
        layout.addWidget(self.chk_tint)

        self.tint_frame = QFrame()
        t_layout = QVBoxLayout(self.tint_frame)
        t_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_tint_amt, self.sl_tint_amt, h_t_amt = make_slider_row("Amount", 0, 100, 20)
        self.sl_tint_amt.valueChanged.connect(lambda: self.shadow_timer.start())
        t_layout.addLayout(h_t_amt)

        t_col_layout = QHBoxLayout()
        t_col_layout.addWidget(QLabel("Colour:"))
        self.btn_tint_color = QPushButton()
        self.btn_tint_color.clicked.connect(self.pick_tint_color)
        self.update_tint_color_button()
        t_col_layout.addWidget(self.btn_tint_color)
        t_col_layout.addStretch()
        t_layout.addLayout(t_col_layout)

        layout.addWidget(self.tint_frame)
        self.tint_frame.hide()

        layout.addSpacing(20)
        layout.addWidget(QLabel("<b>EXPORT SETTINGS</b>"))

        # Format Combo
        fmt_layout = QHBoxLayout()
        fmt_layout.addWidget(QLabel("Format:"))
        self.combo_export_fmt = QComboBox()
        # Mapping display name to internal format keys
        self.combo_export_fmt.addItem("PNG (Lossless)", "png")
        self.combo_export_fmt.addItem("WebP (Lossless)", "webp_lossless")
        self.combo_export_fmt.addItem("WebP (Lossy)", "webp_lossy")
        self.combo_export_fmt.addItem("JPEG (No Transparency)", "jpeg")
        self.combo_export_fmt.currentIndexChanged.connect(self.toggle_export_quality_visibility)
        fmt_layout.addWidget(self.combo_export_fmt)
        layout.addLayout(fmt_layout)

        # Quality Slider
        self.export_quality_frame = QFrame()
        q_layout = QVBoxLayout(self.export_quality_frame)
        q_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_export_quality = QLabel("Quality: 90")
        self.sl_export_quality = QSlider(Qt.Orientation.Horizontal)
        self.sl_export_quality.setRange(1, 100)
        self.sl_export_quality.setValue(90)
        self.sl_export_quality.valueChanged.connect(lambda v: self.lbl_export_quality.setText(f"Quality: {v}"))
        q_layout.addWidget(self.lbl_export_quality)
        q_layout.addWidget(self.sl_export_quality)
        layout.addWidget(self.export_quality_frame)

        # Checkboxes
        self.chk_export_mask = QCheckBox("Save Mask (appends _mask.png)")
        layout.addWidget(self.chk_export_mask)

        self.chk_export_trim = QCheckBox("Trim Transparent Pixels (Auto-Crop)")
        self.chk_export_trim.toggled.connect(self.update_output_preview)
        self.chk_export_trim.setToolTip("If exporting the global mask with the image, the mask is <b>not<b> trimmed. This is to allow you to return to editing the original image.")
        layout.addWidget(self.chk_export_trim)

        # Initialize visibility
        self.toggle_export_quality_visibility()



        layout.addSpacing(40)
        btn_qsave = QPushButton("Quick Save (JPG, White BG)"); btn_qsave.clicked.connect(lambda: self.save_image(quick_save=True))
        layout.addWidget(btn_qsave)
        btn_save = QPushButton("Export Final Image"); btn_save.clicked.connect(self.save_image)
        layout.addWidget(btn_save)
        btn_save_clp = QPushButton("Save to Clipboard"); btn_save_clp.clicked.connect(lambda: self.save_image(clipboard=True))
        layout.addWidget(btn_save_clp)

        

        layout.addStretch()
        scroll.setWidget(container)
        return scroll
    
    def toggle_export_quality_visibility(self):
        """Enables or disables the quality slider based on the selected file format."""
        fmt = self.combo_export_fmt.currentData()
        is_lossy = fmt in ["webp_lossy", "jpeg"]
        self.export_quality_frame.setEnabled(is_lossy)

    def toggle_shadow_options(self, checked):
        if checked: self.shadow_frame.show()
        else: self.shadow_frame.hide()
        self.update_output_preview()

    def toggle_outline_options(self, checked):
        self.outline_frame.setVisible(checked)
        self.update_output_preview()

    def pick_outline_color(self):
        color = QColorDialog.getColor(self.outline_color, self, "Select Outline Colour")
        if color.isValid():
            self.outline_color = color
            self.update_outline_color_button()
            self.update_output_preview()

    def update_outline_color_button(self):
        self.btn_outline_color.setText(self.outline_color.name())
        # Set text colour based on luminance for readability
        text_color = "white" if self.outline_color.lightnessF() < 0.5 else "black"
        self.btn_outline_color.setStyleSheet(
            f"background-color: {self.outline_color.name()}; color: {text_color};"
        )

    def toggle_inner_glow_options(self, checked):
        self.inner_glow_frame.setVisible(checked)
        self.update_output_preview()

    def pick_inner_glow_color(self):
        color = QColorDialog.getColor(self.inner_glow_color, self, "Inner Glow Color")
        if color.isValid():
            self.inner_glow_color = color
            self.update_inner_glow_color_button()
            self.update_output_preview()

    def update_inner_glow_color_button(self):
        text_color = "white" if self.inner_glow_color.lightnessF() < 0.5 else "black"
        self.btn_ig_color.setStyleSheet(f"background-color: {self.inner_glow_color.name()}; color: {text_color};")
        self.btn_ig_color.setText(self.inner_glow_color.name())

    def toggle_tint_options(self, checked):
        self.tint_frame.setVisible(checked)
        self.update_output_preview()

    def pick_tint_color(self):
        color = QColorDialog.getColor(self.tint_color, self, "Tint Color")
        if color.isValid():
            self.tint_color = color
            self.update_tint_color_button()
            self.update_output_preview()

    def update_tint_color_button(self):
        text_color = "white" if self.tint_color.lightnessF() < 0.5 else "black"
        self.btn_tint_color.setStyleSheet(f"background-color: {self.tint_color.name()}; color: {text_color};")
        self.btn_tint_color.setText(self.tint_color.name())

    def create_commit_widget(self):
        self.commit_zone = QFrame()
        self.commit_zone.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        layout = QVBoxLayout(self.commit_zone)
        
        lbl = QLabel("COMMIT MASK TO OUTPUT")
        #lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        h_preview = QHBoxLayout()
        h_preview.setContentsMargins(0, 5, 0, 5)

        self.chk_live_preview = QCheckBox("Live Preview Refine")
        self.chk_live_preview.setToolTip("Automatically run softening/matting etc on the mask for previewing before committing.")
        self.chk_live_preview.toggled.connect(self.handle_live_preview_toggle)
        
        self.sl_live_opacity = QSlider(Qt.Orientation.Horizontal)
        self.sl_live_opacity.setRange(0, 30) 
        self.sl_live_opacity.setValue(30)
        #self.sl_live_opacity.setFixedWidth(70)
        self.sl_live_opacity.setToolTip("Adjust the dimming of the original image background during preview.")
        self.sl_live_opacity.valueChanged.connect(self.show_mask_overlay)
        self.sl_live_opacity.setEnabled(False)

        self.chk_live_preview.toggled.connect(self.sl_live_opacity.setEnabled)

        h_preview.addWidget(self.chk_live_preview)
        h_preview.addStretch()
        h_preview.addWidget(QLabel("Dim:"))
        h_preview.addWidget(self.sl_live_opacity)
        
        layout.addLayout(h_preview)

        h_btns = QHBoxLayout()
        self.btn_add = QPushButton("ADD (A)")
        self.btn_add.clicked.connect(self.add_mask)
        self.btn_add.setToolTip("Add the current model output mask to the composite output image.\n" 
                           "Mask refinement steps e.g. alpha matting are added at this step")
        self.btn_add.setMinimumHeight(50)
        
        self.btn_sub = QPushButton("SUBTRACT (S)")
        self.btn_sub.clicked.connect(self.subtract_mask)
        self.btn_sub.setToolTip("Subtract the current model output mask to the composite output image.\n" 
                           "Mask refinement steps e.g. alpha matting are added at this step")
        self.btn_sub.setMinimumHeight(50)

        self.btn_add.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border: 1px solid #aaa;")
        self.btn_sub.setStyleSheet("background-color: #c62828; color: white; font-weight: bold; border: 1px solid #aaa;")
        
        h_btns.addWidget(self.btn_add)
        h_btns.addWidget(self.btn_sub)
        layout.addLayout(h_btns)
        
        # Global Canvas Actions
        h_canvas = QHBoxLayout()
        btn_in_out = QPushButton("Copy In->Out")
        btn_in_out.clicked.connect(self.copy_input_to_output)
        h_canvas.addWidget(btn_in_out)
        btn_clr_view = QPushButton("Clear View")
        btn_clr_view.clicked.connect(self.clear_visible_area)
        btn_clr_view.setToolTip("Clears the current viewport of the output image, but not the whole image"
                                "\nUseful for refining areas when using automatic models")
        h_canvas.addWidget(btn_clr_view)
        btn_reset_img = QPushButton("Reset Img")
        btn_reset_img.clicked.connect(self.reset_working_image)
        h_canvas.addWidget(btn_reset_img)
        layout.addLayout(h_canvas)

        return self.commit_zone

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
        apply_theme(self, mode)

        self.settings.setValue("theme", mode)


    def toggle_splitter_orientation(self, initial_setup=False):
        current_orientation = self.in_out_splitter.orientation()
        
        if initial_setup:
            target_orientation = current_orientation
        else:
            target_orientation = Qt.Orientation.Vertical if current_orientation == Qt.Orientation.Horizontal else Qt.Orientation.Horizontal
            self.in_out_splitter.setOrientation(target_orientation)

        symbol = "⇳" if target_orientation == Qt.Orientation.Horizontal else "⬄"
        self.in_out_splitter.toggle_button.setText(symbol)

        handle = self.in_out_splitter.handle(1)
        if handle and handle.layout():
            if target_orientation == Qt.Orientation.Horizontal:
                handle.layout().setDirection(QVBoxLayout.Direction.TopToBottom)
            else:
                handle.layout().setDirection(QVBoxLayout.Direction.LeftToRight)

    
    

    
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
            self.status_label.setText(message if message else "Ready")
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
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.load_image_dialog)
        QShortcut(QKeySequence("P"), self).activated.connect(self.chk_paint.toggle)
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self.chk_paint.toggle)
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_image)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(lambda: self.save_image(quick_save=True)) # Quick Save JPG
        QShortcut(QKeySequence("Ctrl+C"), self).activated.connect(lambda: self.save_image(clipboard=True))
        QShortcut(QKeySequence("Ctrl+V"), self).activated.connect(self.load_clipboard)

        QShortcut(QKeySequence("1"), self).activated.connect(lambda: self.tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("2"), self).activated.connect(lambda: self.tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("3"), self).activated.connect(lambda: self.tabs.setCurrentIndex(2))
        QShortcut(QKeySequence("4"), self).activated.connect(lambda: self.tabs.setCurrentIndex(3))
        
        QShortcut(QKeySequence("U"), self).activated.connect(lambda: self.run_automatic_model("u2net"))
        QShortcut(QKeySequence("CTRL+U"), self).activated.connect(lambda: self.run_automatic_model("u2netp"))
        QShortcut(QKeySequence("I"), self).activated.connect(lambda: self.run_automatic_model("isnet-general-use"))
        QShortcut(QKeySequence("O"), self).activated.connect(lambda: self.run_automatic_model("rmbg1_4"))
        QShortcut(QKeySequence("B"), self).activated.connect(lambda: self.run_automatic_model("ben2_base"))
        QShortcut(QKeySequence("M"), self).activated.connect(lambda: self.run_automatic_model("")) # run whatever is selected in the combobox

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
            
            # if image has transparency already, use this as the initial working_mask, otherwise we get a black background
            has_alpha = 'A' in img.mode or (img.mode == 'P' and 'transparency' in img.info)
            initial_mask = None
            if has_alpha:
                alpha = img.getchannel('A')
                # Check if there are actually any transparent pixels. 
                # getextrema() returns (min, max). If min is 255, the alpha is solid white.
                extrema = alpha.getextrema()
                if extrema and extrema[0] < 255:
                    initial_mask = alpha
                else:
                    initial_mask = None

            self.raw_original_image = img.convert("RGBA")

            # NumPy buffer for the adjustment pipeline
            self.adjustment_source_np = np.ascontiguousarray(
                cv2.cvtColor(np.array(self.raw_original_image), cv2.COLOR_RGBA2BGRA)
            )


            self.working_orig_image = self.raw_original_image.copy()

            self.image_exif = img.info.get('exif')
            self.init_working_buffers(initial_mask)

            self.reset_adjustment_sliders()


            self.update_input_view()
            self.update_output_preview()
            self.status_label.setText(f"Loaded: {os.path.basename(path)} [{self.current_image_index + 1}/{len(self.image_paths)}] {'Loaded transparency as global mask' if initial_mask else ''}")
            self.update_window_title()

            self.view_input.set_placeholder(None)
            self.view_output.set_placeholder(None)

            self.thumbnail_strip.blockSignals(True)
            self.thumbnail_strip.setCurrentRow(self.current_image_index)
            self.thumbnail_strip.blockSignals(False)

        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def load_associated_mask(self, image_path):
        mask_path = os.path.splitext(image_path)[0] + "_nobg_mask.png"
        if os.path.exists(mask_path):
            try:
                mask = Image.open(mask_path).convert("L")
                if mask.size == self.working_orig_image.size:
                    self.add_undo_step()
                    self.working_mask = mask
                    self.update_output_preview()
                    self.status_label.setText(f"Loaded associated mask: {os.path.basename(mask_path)}")
                else:
                     QMessageBox.warning(self, "Mask Size Mismatch", "The associated mask's dimensions do not match the base image.")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Mask", f"Could not load the associated mask:\n{e}")
    
    def load_associated_trimap(self, image_path):
        """
        Looks for a file named [image_name]_trimap.png and loads it into 
        the custom trimap buffer without enabling alpha matting.
        """
        trimap_path = os.path.splitext(image_path)[0] + "_trimap.png"
        if os.path.exists(trimap_path):
            try:
                trimap = Image.open(trimap_path).convert("L")
                if trimap.size == self.working_orig_image.size:
                    self.user_trimap = trimap
                    
                    # Set UI to Custom mode but don't enable alpha matting or display yet
                    # This is so it doesnt get overwritten by automatic trimap editing
                    self.rb_trimap_custom.blockSignals(True)
                    self.rb_trimap_custom.setChecked(True)
                    self.rb_trimap_custom.blockSignals(False)
                    
                    self.status_label.setText(f"Loaded associated trimap: {os.path.basename(trimap_path)}")
                else:
                    print(f"Trimap error: size mismatch for {trimap_path}")
            except Exception as e:
                print(f"Error loading associated trimap: {e}")

    def load_blank_image(self):
        self.working_orig_image = Image.new("RGBA", (800, 600), (0,0,0,0))
        self.init_working_buffers()
        self.update_input_view()

    def load_clipboard(self):
        img = ImageGrab.grabclipboard()

        if isinstance(img, Image.Image):

            has_alpha = 'A' in img.mode or (img.mode == 'P' and 'transparency' in img.info)
            initial_mask = None
            if has_alpha:
                alpha = img.getchannel('A')
                # Check if there are actually any transparent pixels. 
                # getextrema() returns (min, max). If min is 255, the alpha is solid white.
                extrema = alpha.getextrema()
                if extrema and extrema[0] < 255:
                    initial_mask = alpha
                else:
                    initial_mask = None

            self.raw_original_image = img.convert("RGBA")
            self.adjustment_source_np = np.ascontiguousarray(
                cv2.cvtColor(np.array(self.raw_original_image), cv2.COLOR_RGBA2BGRA)
            )
            self.working_orig_image = self.raw_original_image.copy()

            self.image_paths = ["Clipboard"]
            self.init_working_buffers(initial_mask)
            self.reset_adjustment_sliders()
            self.update_input_view()
            self.update_output_preview()
            self.update_thumbnail_strip()
            self.status_label.setText("Loaded from Clipboard [1/1]")
            self.update_window_title()

            self.view_input.set_placeholder(None)
            self.view_output.set_placeholder(None)
            
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
            if mask.size == self.working_orig_image.size:
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

    def init_working_buffers(self, initial_mask=None):
        size = self.working_orig_image.size 
        if initial_mask:
            self.working_mask = initial_mask.convert("L").copy()
            self.status_label.setText("Loaded image transparency as global mask")
        else:
            self.working_mask = Image.new("L", size, 0)

        self.model_output_mask = Image.new("L", size, 0)
        self.refined_preview_mask = None
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

    def update_input_view(self, reset_zoom=True):
        if self.working_orig_image:
            # 1. Update the Pixmap
            self.input_pixmap_item.setPixmap(pil2pixmap(self.working_orig_image))
            rect = self.input_pixmap_item.boundingRect()
            self.view_input.setSceneRect(rect)

            if reset_zoom:
                self.view_input.resetTransform()
                self.view_input.fitInView(self.input_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
                # Set to fill the screen, but could add small padding if preferred
                self.view_input.scale(1, 1)

            if self.view_output:
                self.view_output.setSceneRect(rect)
                if reset_zoom:
                    self.view_output.setTransform(self.view_input.transform())
            
            self.update_zoom_label()

    def update_zoom_label(self):
        zoom = self.view_input.transform().m11() * 100
        self.zoom_label.setText(f"Zoom: {int(zoom)}%")

    def eventFilter(self, source, event):
        """Captures mouse entry into views to display control hints in the status bar."""
        if event.type() == QEvent.Type.Enter:
            if not self.is_busy():
                if self.paint_mode:
                    msg = VIEW_PAINT_MSG
                elif source == self.view_input.viewport():
                    msg = VIEW_IN_MSG
                elif source == self.view_output.viewport():
                    msg = VIEW_OUT_MSG
                self.status_label.setText(msg)
        elif event.type() == QEvent.Type.Leave:
            if not self.is_busy():
                self.status_label.setText("Ready")

        return super().eventFilter(source, event)

    def get_viewport_crop(self):
        vp = self.view_input.viewport().rect()
        sr = self.view_input.mapToScene(vp).boundingRect()
        ir = QRectF(0, 0, self.working_orig_image.width, self.working_orig_image.height)
        cr = sr.intersected(ir)
        x = int(round(cr.x()))
        y = int(round(cr.y()))
        w = int(round(cr.width()))
        h = int(round(cr.height()))
        if w <= 0 or h <= 0: return self.working_orig_image, 0, 0
        return self.working_orig_image.crop((x, y, x+w, y+h)), x, y

    

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
            self.model_output_mask = Image.new("L", self.working_orig_image.size, 0)
            self.overlay_pixmap_item.setPixmap(QPixmap())
            self.status_label.setText("Ready (No points in view)")
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
        
        # clear any previous mask or sam points
        # ideally would be after inference, but on_inference_finished is shared by sam and auto models
        # self.clear_overlay()

        use_2step = self.chk_2step_auto.isChecked()
        
        if use_2step:
            # use the full size original image first
            image_to_process = self.working_orig_image
            x_off, y_off = 0, 0
        else:
            image_to_process, x_off, y_off = self.get_viewport_crop()

        if image_to_process.width == 0 or image_to_process.height == 0:
            return

        provider_data = self.combo_auto_model_EP.currentData()
        self.set_loading(True, f"Processing {model_name}...")

        def _do_auto_work(model_manager, model_name, provider_data, img, is_2step):
            prov_str, prov_opts, prov_code = provider_data
            session, load_t = model_manager.get_auto_session(model_name, provider_data)
            # Send prov_code to generate the status message
            if is_2step:
                mask_arr, status = model_manager.run_auto_inference_2step(session, img, model_name, load_t, prov_code)
            else:
                mask_arr, status = model_manager.run_auto_inference(session, img, model_name, load_t, prov_code)
                
            return {"mask": mask_arr, "status": status}

        self.worker = InferenceWorker(_do_auto_work, 
                                      self.model_manager, 
                                      model_name, 
                                      provider_data, 
                                      image_to_process, use_2step)
        self.worker.finished.connect(self.clear_overlay)
        self.worker.finished.connect(lambda res: self._on_inference_finished(res, x_off, y_off))
        self.worker.error.connect(lambda msg: (self.set_loading(False), QMessageBox.critical(self, "Error", msg)))
        self.worker.start()

    def _on_inference_finished(self, result, x_off, y_off):
        """Processes model output masks into a overlay, and updates UI"""
        mask_arr = result["mask"]
        status_msg = result["status"]

        # Paste the viewport mask into the correct place
        self.model_output_mask = Image.new("L", self.working_orig_image.size, 0)
        self.model_output_mask.paste(Image.fromarray(mask_arr, mode="L"), (x_off, y_off))
        self.refined_preview_mask = None
        
        # Update UI
        if self.chk_live_preview.isChecked():
            # Automatically start the refinement thread
            self.modify_mask(op="live_preview")
        else:
            self.show_mask_overlay()
        self.update_cached_model_icons()
        
        self.set_loading(False, status_msg)

        if hasattr(self, 'worker'):
            self.worker.deleteLater()
            self.worker = None


    def show_mask_overlay(self):
        is_live = self.chk_live_preview.isChecked()
        mask_only = self.chk_input_mask_only.isChecked()

        live_preview_alpha = self.sl_live_opacity.value() / 100.0
        
        # Determine which mask to display
        active_mask = self.refined_preview_mask if is_live and self.refined_preview_mask else self.model_output_mask

        if is_live and self.refined_preview_mask:
            # live preview logic
            self.input_pixmap_item.show()
            
            if mask_only:
                # show the high-res alpha mask in grayscale
                self.input_pixmap_item.setOpacity(0.0)
                self.overlay_pixmap_item.setOpacity(1.0)
                self.overlay_pixmap_item.setPixmap(pil2pixmap(active_mask))
            else:
                # image cutout over dimmed background
                self.input_pixmap_item.setOpacity(live_preview_alpha) # Dim the original
                self.overlay_pixmap_item.setOpacity(1.0)
                
                cutout = self.working_orig_image.convert("RGBA")
                cutout.putalpha(active_mask)
                self.overlay_pixmap_item.setPixmap(pil2pixmap(cutout))
        else:
            # original blue overlay mode
            self.input_pixmap_item.setOpacity(1.0)
            
            if mask_only:
                self.overlay_pixmap_item.setOpacity(1.0)
                self.overlay_pixmap_item.setPixmap(pil2pixmap(active_mask))
                self.input_pixmap_item.hide()
            else:
                # Show blue overlay on top of original image
                self.overlay_pixmap_item.setOpacity(0.5)
                self.input_pixmap_item.show()
                blue = Image.new("RGB", self.working_orig_image.size, (0, 0, 255))
                overlay = blue.convert("RGBA")
                overlay.putalpha(active_mask)
                self.overlay_pixmap_item.setPixmap(pil2pixmap(overlay))
        
        self.update_trimap_preview()

    def clear_overlay(self):
        self.coordinates = []
        self.labels = []
        self.model_output_mask = Image.new("L", self.working_orig_image.size, 0)
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

            if item == self.view_input.crop_rect_item: continue
            
            # Delete SAM points/boxes
            if isinstance(item, (QGraphicsEllipseItem, QGraphicsRectItem, QGraphicsPathItem)):
                items_to_remove.append(item)
                
        for item in items_to_remove:
            self.scene_input.removeItem(item)

    def clear_visible_area(self):
        if not self.working_mask: 
            return
            
        self.add_undo_step()
        
        # Get the exact pixel boundaries currently in view
        crop, x, y = self.get_viewport_crop()
        
        # Create a black rectangle of the viewport size
        # We use a solid 0-value (black) to 'clear' the mask
        clear_rect = Image.new("L", (crop.width, crop.height), 0)
        
        # Paste directly into the working mask at the viewport offset
        # PIL's paste uses (left, top, right, bottom) or (left, top)
        # Using (x, y) coordinates derived from the floor/ceil logic above
        self.working_mask.paste(clear_rect, (x, y))
        
        self.update_output_preview()
        self.status_label.setText(f"Cleared viewport area: {crop.width}x{crop.height}") 

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
        
        self.auto_trimap_sliders_widget.setEnabled(is_auto)
        
        self.btn_edit_trimap.setEnabled(not is_auto)
        
        self.update_trimap_preview()

    def open_trimap_editor(self):
        if not self.working_orig_image:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        initial_trimap = None

        if hasattr(self, 'user_trimap') and self.user_trimap:
            initial_trimap = self.user_trimap
        # Otherwise, generate one from the current mask and sliders as a starting point.
        elif self.model_output_mask:
            fg_erode = self.sl_fg_erode.value()
            bg_erode = self.sl_bg_erode.value()
            trimap_np = generate_trimap_from_mask(self.model_output_mask, fg_erode, bg_erode)
            initial_trimap = Image.fromarray(trimap_np)
        else:
            # If there's no mask at all, start with a blank (all unknown) trimap.
            initial_trimap = Image.new("L", self.working_orig_image.size, 128)

        dialog = TrimapEditorDialog(self.working_orig_image, initial_trimap, self)
        if dialog.exec():
            # If the user clicked OK, store the result
            self.user_trimap = dialog.final_trimap
            self.last_trimap = np.array(self.user_trimap) 
            
            self.rb_trimap_custom.setChecked(True)
            
            # Ensure the preview is shown
            self.chk_show_trimap.setChecked(True)
            self.trigger_refinement_update()
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
            # only show trimap immediately when live preview is off, otherwise difficult to a/b test
            if not self.chk_live_preview.isChecked():
                self.chk_show_trimap.setChecked(True)
        else:
            # If unchecked, also uncheck the trimap view so it's off if re-enabled
            self.chk_show_trimap.setChecked(False)
            self.view_input.hide_legend()

    def toggle_trimap_display(self, checked):
        """Shows or hides the trimap overlay on the input view."""
        if checked:
            self.update_trimap_preview()
            self.trimap_overlay_item.show()
        else:
            self.trimap_overlay_item.hide()
            self.view_input.hide_legend()


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
            trimap_np = generate_trimap_from_mask(self.model_output_mask, fg_erode, bg_erode)
        
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
        apply_binary = self.chk_binarise_mask.isChecked()

        msg = " Alpha matting can take a while" if apply_matting else ""
        self.set_loading(True, "Applying mask..." + msg)
        
        matting_params = {}
        if apply_matting:
            try:
                # Unsure if the normal logic of cropping the image to the viewport is as relevant when matting models
                # are resolution agnostic. But for performance on my machine, and consistency with automatic models,
                # will keep matting to the viewport
                image_crop, x_off, y_off = self.get_viewport_crop()
                trimap_np = None
                limit = int(self.settings.value("matting_longest_edge", 1024))

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
                    trimap_np = generate_trimap_from_mask(mask_crop, fg_erode, bg_erode)
                
                matting_params = {
                    'image_crop': image_crop,
                    'trimap_np': trimap_np,
                    'x_off': x_off, 'y_off': y_off,
                    'algorithm': self.combo_matting_algorithm.currentText(),
                    # For now use the provider we have selected in automatic models combobox
                    # Unsure if worth giving user the option to select EP, since VitMatte is essentially a automatic model
                    'provider_data': self.combo_auto_model_EP.currentData(),
                    'longest_edge_limit': limit
                }
            except Exception as e:
                QMessageBox.critical(self, "Alpha Matting Prep Error", str(e))
                self.set_loading(False)
                return

        def _do_modify_work(model_manager, base_mask, matting_enabled, soften_enabled, binary_enabled, m_params):
            processed_mask = base_mask

            if matting_enabled and m_params:
                # Construct settings for the crop matting step
                mat_settings = {
                    "matting": {
                        "enabled": True,
                        "algorithm": m_params['algorithm'],
                        "provider_data": m_params['provider_data'],
                        "longest_edge_limit": m_params['longest_edge_limit']
                    }
                }
                
                # Extract mask crop to match the image crop
                x, y = m_params['x_off'], m_params['y_off']
                w, h = m_params['image_crop'].size
                mask_crop = processed_mask.crop((x, y, x + w, y + h))
                
                # Run Refine Mask on the crop (using passed trimap)
                matted_alpha_crop = refine_mask(
                    mask_crop, 
                    m_params['image_crop'], 
                    mat_settings, 
                    model_manager, 
                    trimap_np=m_params['trimap_np']
                )

                if matted_alpha_crop:
                    # Create a new mask to avoid modifying the original in the thread
                    new_m = processed_mask.copy()
                    # Create a black patch to clear the area under the new matted crop
                    paste_area = Image.new("L", matted_alpha_crop.size, 0)
                    new_m.paste(paste_area, (x, y))
                    # Paste the new matted result
                    new_m.paste(matted_alpha_crop, (x, y))
                    processed_mask = new_m

            # Apply Global Effects (Soften / Binarise)
            global_settings = {
                "soften": soften_enabled,
                "soften_radius": SOFTEN_RADIUS,
                "binarise": binary_enabled
            }
            
            # Since matting is already done (or disabled), we don't need to pass image/model_manager here
            # Should probably split into two functions instead of using different parts of one a second time...
            processed_mask = refine_mask(processed_mask, None, global_settings, None)
            
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

        if not op == "live_preview":
            # commit refined mask to working_mask
            self.working_mask = op(self.working_mask, processed_mask)
            self.update_output_preview()
        else:
            # show live preview
            self.refined_preview_mask = processed_mask
            self.show_mask_overlay()
        self.set_loading(False, "Ready")

    

    

    def toggle_shadow_options(self, checked):
        if checked: self.shadow_frame.show()
        else: self.shadow_frame.hide()
        self.update_output_preview()

    

    #@profile
    def render_output_image(self, shadow_downscale=0.125):
        if not self.working_orig_image: return

        # Get Settings
        settings = self.get_render_settings()
        
        # Inject downscale for GUI performance (not in base settings as batch might want full quality)
        settings['shadow']['downscale'] = shadow_downscale

        current_mask = self.working_mask
        if settings["clean_alpha"]:
            current_mask = clean_alpha(current_mask)

        # Handle Caching Logic for Foreground Estimation
        precomputed_cutout = None
        
        # Check if we need to update the cached cutout
        adj_params = self.get_adjustment_params()
        adj_hash = hash(frozenset(adj_params.items()))
        
        if settings["foreground_correction"]["enabled"]:
             try:
                current_mask_hash = hash(current_mask.tobytes())
                fg_algo = settings["foreground_correction"]["algorithm"]
                current_state_key = (current_mask_hash, adj_hash, fg_algo)

                if current_state_key != self.working_mask_hash:
                    self.set_loading(True, f"Estimating foreground colour correction ({fg_algo.upper()})")
                    # We compute it here to cache it on the instance
                    precomputed_cutout = self.model_manager.estimate_foreground(self.working_orig_image, current_mask, fg_algo)
                    self.set_loading(False)
                    
                    self.working_mask_hash = current_state_key
                    self.cached_fg_corrected_cutout = precomputed_cutout
                else:
                    precomputed_cutout = self.cached_fg_corrected_cutout
             except Exception as e:
                print(f"Error during foreground estimation: {e}")

        # Handle Caching Logic for Blurred Background
        precomputed_bg = None
        if "Blur" in settings["background"]["type"]:
             mask_hash = hash(current_mask.tobytes())
             rad = settings["background"]["blur_radius"]
             current_params = (mask_hash, adj_hash, rad)
             
             if current_params == self.last_blur_params and self.cached_blurred_bg is not None:
                 precomputed_bg = self.cached_blurred_bg
             else:
                 self.set_loading(True, "Blurring Background")
                 precomputed_bg = generate_blurred_background(self.working_orig_image, current_mask, rad)
                 self.set_loading(False)
                 
                 self.cached_blurred_bg = precomputed_bg
                 self.last_blur_params = current_params


        final = compose_final_image(
            self.working_orig_image, 
            self.working_mask, 
            settings, 
            model_manager=self.model_manager,
            precomputed_cutout=precomputed_cutout,
            precomputed_bg=precomputed_bg
        )
        
        self.last_render = final.copy()
        return final

    

    def update_output_preview(self):

        if self.chk_show_mask.isChecked(): 
            
            mask = self.working_mask
            if self.chk_clean_alpha.isChecked():
                mask = clean_alpha(mask)

            if self.chk_show_partial_alpha.isChecked():
                final = generate_alpha_map(mask)
            else:
                final = mask.convert("RGBA")
        
        else:
            final = self.render_output_image()
        
        self.output_pixmap_item.setPixmap(pil2pixmap(final))
        self.view_output.setSceneRect(self.view_input.sceneRect())

        # Update the Auto-Trim visual overlay
        if self.chk_export_trim.isChecked() and not self.chk_show_mask.isChecked():
            bbox = get_current_crop_bbox(self.working_mask, self.chk_shadow.isChecked(),
                                         self.sl_s_x.value(),
                                         self.sl_s_y.value(),
                                         self.sl_s_r.value())
            if bbox:
                # Create a path for the whole scene
                full_rect = self.output_pixmap_item.boundingRect()
                path = QPainterPath()
                path.addRect(full_rect)
                
                # Create the "hole" for the crop area
                crop_rect = QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                
                # Subtracting the crop_rect from the full_rect path creates the dimmed border effect
                # We use the 'subtracted' method to ensure a clean cutout
                hole_path = QPainterPath()
                hole_path.addRect(crop_rect)
                
                final_path = path.subtracted(hole_path)
                self.output_crop_overlay.setPath(final_path)
                self.output_crop_overlay.show()
            else:
                self.output_crop_overlay.hide()
        else:
            self.output_crop_overlay.hide()

    def reset_all(self):
        self.clear_overlay()

        # Clear Trimap and reset UI
        self.last_trimap = None
        if hasattr(self, 'user_trimap'):
            del self.user_trimap
        self.rb_trimap_auto.setChecked(True)

        self.add_undo_step()
        self.working_mask = Image.new("L", self.working_orig_image.size, 0)
        self.update_output_preview()
        
    def reset_working_image(self):
        self.add_undo_step()
        self.working_mask = Image.new("L", self.working_orig_image.size, 0)
        self.update_output_preview()
        
    def copy_input_to_output(self):
        if self.working_orig_image:
            self.add_undo_step()
            # Use the alpha channel of the image as the mask (e.g. PNG files)
            # JPG will gain an alpha of 255 when converted to RGBA in load_image
            self.working_mask = self.working_orig_image.split()[3]
        else:
            self.add_undo_step()
            self.working_mask = Image.new("L", self.working_orig_image.size, 255)
        
        self.update_output_preview()

    def toggle_paint_mode(self, enabled):
        if enabled and self.crop_mode:
            self.btn_toggle_crop.setChecked(False)
        
        self.paint_mode = enabled
        self.view_input.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.view_output.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        
        # Update cursor on both views
        cursor_pos = self.view_input.mapFromGlobal(QCursor.pos())
        scene_pos = self.view_input.mapToScene(cursor_pos)
        self.view_input.update_brush_cursor(scene_pos)
        self.view_output.update_brush_cursor(scene_pos)

        if not self.is_busy():
            if self.view_input.underMouse():
                if enabled:
                    self.status_label.setText(VIEW_PAINT_MSG)
                else:
                    self.status_label.setText(VIEW_IN_MSG)

    def handle_paint_start(self, pos):
        
        if self.is_busy():
            return # Interaction blocked during inference
        
        # If a path is already being drawn, do nothing.
        if hasattr(self, 'current_path'): return

        modifiers = QApplication.keyboardModifiers()
        buttons = QApplication.mouseButtons()
        self.is_erasing = bool(buttons & Qt.MouseButton.RightButton)
        self.is_smart_refine = bool(modifiers & Qt.KeyboardModifier.ControlModifier) and not self.is_erasing
        
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
        if self.is_smart_refine:
            color = QColor(0, 255, 255, 200) # Cyan
        else:
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

            if self.is_smart_refine:
                coords = np.column_stack(np.where(stroke_mask_np > 0))
                if coords.size == 0: return
                
                # padding to ensure the AI sees enough 'definite' FG/BG pixels
                padding = int(self.brush_width * self.sl_smart_padding.value())

                y1, x1 = np.min(coords, axis=0) - padding
                y2, x2 = np.max(coords, axis=0) + padding
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                image_patch = self.working_orig_image.crop((x1, y1, x2, y2))
                mask_patch_np = np.array(self.working_mask.crop((x1, y1, x2, y2)))
                local_stroke_np = stroke_mask_np[y1:y2, x1:x2]

                # 128 for brush area. Anything over 220 assi,e os foreground
                trimap_np = np.where(mask_patch_np > 220, 255, 0).astype(np.uint8)
                trimap_np[local_stroke_np > 0] = 128
                
                self.set_loading(True, "Smart Refining...")
                
                limit = int(self.settings.value("matting_longest_edge", 1024))
                matting_params = {
                    'image_crop': image_patch,
                    'trimap_np': trimap_np,
                    'x_off': x1, 'y_off': y1,
                    'algorithm': self.combo_matting_algorithm.currentText(),
                    'provider_data': self.combo_auto_model_EP.currentData(),
                    'longest_edge_limit': limit,
                    'local_stroke_np': local_stroke_np # Pass this to the handler
                }

                def _do_smart_refine_work(model_manager, m_params):
                    refined_patch = model_manager.run_matting(
                        m_params['algorithm'], 
                        m_params['image_crop'], 
                        m_params['trimap_np'], 
                        m_params['provider_data'],
                        m_params['longest_edge_limit']
                    )
                    return refined_patch, m_params

                self.worker = InferenceWorker(_do_smart_refine_work, self.model_manager, matting_params)
                self.worker.finished.connect(self._on_smart_refine_finished)
                self.worker.start()
                return 

            if self.chk_soften.isChecked():
                # ksize (0,0) allows OpenCV to compute the kernel size automatically from sigma
                stroke_mask_np = cv2.GaussianBlur(stroke_mask_np, (0, 0), sigmaX=SOFTEN_RADIUS)
            
            stroke_mask = Image.fromarray(stroke_mask_np, mode="L")

            # Check preference for paint target
            edits_working_directly = self.settings.value("paint_edits_working_mask", True, type=bool)

            if edits_working_directly:
                self.add_undo_step()
                if self.is_erasing:
                    # Right Click: Remove from composite
                    self.working_mask = ImageChops.subtract(self.working_mask, stroke_mask)
                else:
                    # Left Click: Add to composite (Paint from original image)
                    self.working_mask = ImageChops.add(self.working_mask, stroke_mask)
                
                self.update_output_preview()
            else:
                # Legacy behaviour: Edit the model output mask (blue overlay)
                # This requires user to click 'Add' or 'Subtract' to commit to composite
                if self.is_erasing:
                    self.model_output_mask = ImageChops.subtract(self.model_output_mask, stroke_mask)
                else:
                    self.model_output_mask = ImageChops.add(self.model_output_mask, stroke_mask)
                self.show_mask_overlay()
            
            self.update_output_preview()
            
        finally:
            if hasattr(self, 'current_path'): 
                del self.current_path
            QApplication.restoreOverrideCursor()

    def _on_smart_refine_finished(self, result):
        refined_alpha, m_params = result
        x, y = m_params['x_off'], m_params['y_off']
        local_stroke_np = m_params['local_stroke_np']

        feather_radius = max(3, int(self.brush_width * 0.3))
        if feather_radius % 2 == 0:
            feather_radius += 1
            
        # Apply blur to the stroke patch to create the soft transition
        soft_stencil_np = cv2.stackBlur(local_stroke_np, (feather_radius, feather_radius), 0)
        stencil = Image.fromarray(soft_stencil_np, mode="L")

        self.add_undo_step()
        
        self.working_mask.paste(refined_alpha, (x, y), mask=stencil)
        
        self.set_loading(False, "Smart Refine applied to stroke.")
        self.update_output_preview()

    def save_image(self, quick_save=False, clipboard = False):
        if not self.image_paths: return
        
        fmt = self.combo_export_fmt.currentData()
        quality = self.sl_export_quality.value()
        save_mask = self.chk_export_mask.isChecked()
        trim = self.chk_export_trim.isChecked()
        
        ext_map = {"png": "png", "webp_lossless": "webp", "webp_lossy": "webp", "jpeg": "jpg"}
        default_ext = ext_map[fmt]
        
        if quick_save:
            default_ext= "jpg"
            quality = 95
            fmt = "jpeg"

        if not clipboard:
            initial_name = os.path.splitext(self.image_paths[0])[0] + "_nobg." + default_ext
            initial_name = sanitise_filename_for_windows(initial_name)

            fname, _ = QFileDialog.getSaveFileName(self, "Export Image", initial_name, f"{default_ext.upper()} (*.{default_ext})")
            if not fname: return

            fname = sanitise_filename_for_windows(fname)
            
            if not fname.lower().endswith(f".{default_ext}"): fname += f".{default_ext}"

        self.set_loading(True, "Exporting...")
        # use cached generated image for speed
        final_image = self.last_render

        # Zero out RGB data in transparent areas for export.
        # Do this now instead of in render_output_image to speed up display time
        # This prevents "ghost backgrounds" when re-loading the file.
        # These ghost backgrounds can however make some interesting unexpected backgrounds!
        clean_canvas = Image.new("RGBA", final_image.size, (0, 0, 0, 0))
        clean_canvas.paste(final_image, (0, 0), final_image)
        final_image = clean_canvas

        # If trimming is enabled, calculate the crop box and apply it.
        if trim:
            bbox = get_current_crop_bbox(self.working_mask, self.chk_shadow.isChecked(), self.sl_s_x.value(), self.sl_s_y.value(), self.sl_s_r.value())
            if bbox:
                final_image = final_image.crop(bbox)

        if clipboard:
            self.save_image_clipboard(final_image)
            self.set_loading(False,"Saved to Clipboard")
            return

        if fmt == "jpeg":
            background = Image.new("RGB", final_image.size, (255, 255, 255))
            # Ensure the image has an alpha channel to use as a mask
            if final_image.mode != 'RGBA':
                final_image = final_image.convert('RGBA')
            background.paste(final_image, mask=final_image.split()[3])
            final_image = background
        
        save_params = {}
        if self.image_exif: save_params['exif'] = self.image_exif
        if fmt == "jpeg": save_params['quality'] = quality
        elif fmt == "webp_lossy": save_params['quality'] = quality
        elif fmt == "webp_lossless": save_params['lossless'] = True
        elif fmt == "png": save_params['optimize'] = True

        try:
            final_image.save(fname, **save_params)
            lbl = f"Saved to {fname}"
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            
        if save_mask:
            mname = os.path.splitext(fname)[0] + "_mask.png"
            mask = self.working_mask
            if self.chk_clean_alpha.isChecked():
                mask = clean_alpha(mask)
            mask.save(mname)
            lbl = f"Saved to {os.path.basename(fname)} and {os.path.basename(mname)}"

        self.set_loading(False,lbl)

    def save_image_clipboard(self, final_image):
        if not self.image_paths: return

        data = final_image.tobytes("raw", "RGBA")
        qimage = QImage(
            data,
            final_image.width,
            final_image.height,
            QImage.Format.Format_RGBA8888
        ).copy()

        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        qimage.save(buffer, "PNG")
        png_data = buffer.data()

        mime_data = QMimeData()
        mime_data.setData("image/png", png_data)
        mime_data.setData("PNG", png_data)
        mime_data.setImageData(qimage)

        QGuiApplication.clipboard().setMimeData(mime_data)


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
- P: Toggle Paintbrush (draw manually on mask). Left click - Add, Right click - Erase, Ctrl+Left Click - Smart Refine

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
- M: Run automatic model selected in the menu
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
    parser.add_argument("--load-trimap", action="store_true", help="Load associated _trimap.png file if present")
    parser.add_argument("--bg-colour", type=str)
    parser.add_argument("--binary", action="store_true", help="Start with Binary Mask enabled")
    parser.add_argument("--soften", action="store_true", help="Start with Soften Mask enabled")
    parser.add_argument("--shadow", action="store_true", help="Start with Drop Shadow enabled")
    parser.add_argument("--colour-correction", action="store_true", help="Start with Colour Correction enabled")
    parser.add_argument("--load-mask", action="store_true", help="Load associated _mask.png file if present")



    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Interactive Background Remover")
    app.setDesktopFileName("Interactive Background Remover")
    # Set fusion for consistency across OS. The theme will be loaded from settings.
    app.setStyle("Fusion")
    window = BackgroundRemoverGUI(args.images, args)
    window.showMaximized()
    sys.exit(app.exec())