#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import cv2
import gc
from timeit import default_timer as timer

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QSplitter, QDialog, QScrollArea, 
                             QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem, 
                             QTextEdit, QSizePolicy, QRadioButton, QButtonGroup, QInputDialog, 
                             QProgressBar, QStyle)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (QPixmap, QImage, QColor, QPainter, QPainterPath, QPen, QBrush,
                         QKeySequence, QShortcut, QCursor, QIcon)

from PIL import Image, ImageOps, ImageDraw, ImageEnhance, ImageGrab, ImageFilter, ImageChops
import onnxruntime as ort

import download_manager

# --- Constants ---
DEFAULT_ZOOM_FACTOR = 1.15
if getattr(sys, 'frozen', False):
    SCRIPT_BASE_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ROOT_DIR = os.path.join(SCRIPT_BASE_DIR, "Models/")

CACHE_ROOT_DIR = os.path.join(SCRIPT_BASE_DIR, "Models", "cache")

PAINT_BRUSH_SCREEN_SIZE = 30 
UNDO_STEPS = 20
SOFTEN_RADIUS = 2 
MIN_RECT_SIZE = 10


SAM_TRT_WARMUP_POINTS = 30  # how many interactive points to pre-compile for TensorRT


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


def pil2pixmap(im):
    if im is None: return QPixmap()
    
    if im.mode == "RGB":
        im = im.convert("RGBA")
    elif im.mode == "L":
        im = im.convert("RGBA")

    if isinstance(im, np.ndarray):
        img_data = im
        height, width, channels = img_data.shape
    else:
        img_data = np.array(im, copy=False)
        width, height = im.size
        
    bytes_per_line = width * 4 
    
    qim = QImage(img_data.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)

    return QPixmap.fromImage(qim)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SaveOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Options")
        self.setFixedSize(350, 300)
        
        layout = QVBoxLayout(self)
        
        # Format Selection
        grp = QFrame(); v_grp = QVBoxLayout(grp)
        v_grp.addWidget(QLabel("<b>File Format:</b>"))
        
        self.rb_png = QRadioButton("PNG")
        self.rb_webp_lossless = QRadioButton("WebP (Lossless)")
        self.rb_webp_lossy = QRadioButton("WebP (Lossy)")
        self.rb_jpg = QRadioButton("JPEG")
        
        self.bg = QButtonGroup()
        self.bg.addButton(self.rb_png, 0)
        self.bg.addButton(self.rb_webp_lossless, 1)
        self.bg.addButton(self.rb_webp_lossy, 2)
        self.bg.addButton(self.rb_jpg, 3)
        self.rb_png.setChecked(True)
        
        v_grp.addWidget(self.rb_png)
        v_grp.addWidget(self.rb_webp_lossless)
        v_grp.addWidget(self.rb_webp_lossy)
        v_grp.addWidget(self.rb_jpg)
        layout.addWidget(grp)
        
        # Quality Slider
        self.q_frame = QFrame()
        q_layout = QVBoxLayout(self.q_frame)
        self.lbl_quality = QLabel("Quality: 90")
        self.sl_quality = QSlider(Qt.Orientation.Horizontal)
        self.sl_quality.setRange(1, 100)
        self.sl_quality.setValue(90)
        self.sl_quality.valueChanged.connect(lambda v: self.lbl_quality.setText(f"Quality: {v}"))
        
        q_layout.addWidget(self.lbl_quality)
        q_layout.addWidget(self.sl_quality)
        layout.addWidget(self.q_frame)
        
        # Save Mask Checkbox
        self.chk_mask = QCheckBox("Save Mask (appends _mask.png)")
        layout.addWidget(self.chk_mask)

        self.chk_trim = QCheckBox("Trim Transparent Pixels (Auto-Crop)")
        layout.addWidget(self.chk_trim)
        
        # Logic to enable/disable quality slider
        self.bg.idToggled.connect(self.toggle_quality)
        self.toggle_quality() # Init state

        # Buttons
        btns = QHBoxLayout()
        b_ok = QPushButton("OK"); b_ok.clicked.connect(self.accept)
        b_cancel = QPushButton("Cancel"); b_cancel.clicked.connect(self.reject)
        btns.addWidget(b_ok); btns.addWidget(b_cancel)
        layout.addLayout(btns)

    def toggle_quality(self):
        # ID 2 is WebP Lossy, 3 is JPEG
        is_lossy = self.bg.checkedId() in [2, 3]
        self.q_frame.setEnabled(is_lossy)

    def get_data(self):
        fmt_map = {0: "png", 1: "webp_lossless", 2: "webp_lossy", 3: "jpeg"}
        return {
            "format": fmt_map[self.bg.checkedId()],
            "quality": self.sl_quality.value(),
            "save_mask": self.chk_mask.isChecked(),
            "trim": self.chk_trim.isChecked()
        }

# Animated Collapsable Widget
class CollapsibleFrame(QFrame):
    def __init__(self, title="Options", parent=None, animation_duration=250, tooltip=None):
        super().__init__(parent)
        self.animation_duration = animation_duration
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0) # No margin on the outer frame
        self.main_layout.setSpacing(0)
        
        # 1. Header Button
        self.toggle_button = QPushButton(title)
        if tooltip:
            self.toggle_button.setToolTip(tooltip)
        self.toggle_button.setStyleSheet("text-align: left; padding: 5px;")
        self.toggle_button.setFlat(True)
        self.toggle_button.clicked.connect(self.toggle_content)
        
        # 2. Content Frame
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.setSpacing(5)
        
        # 3. Animation Setup
        self.animation = QPropertyAnimation(self.content_frame, b"maximumHeight")
        self.animation.setDuration(self.animation_duration)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # 4. Add to Main Layout
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_frame)
        
        # Set initial state to collapsed (but hide/show the contents)
        self.is_collapsed = True
        self.content_frame.setVisible(False)
        self.content_frame.setMaximumHeight(0)
        self.toggle_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarUnshadeButton))

    def layout_for_content(self):
        """Returns the inner layout where the user's widgets should be added."""
        return self.content_layout
            
    def toggle_content(self, checked=None):
        self.animation.stop()

        if self.is_collapsed:
            self.is_collapsed = False
            self.content_frame.setVisible(True) 
            self.toggle_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarShadeButton))
            
            collapsed_height = self.content_frame.height()
            self.content_frame.setMaximumHeight(collapsed_height)
            
            # Set the end value to a very large number, 
            # and the layout manager will clamp it to the required size.
            target_height = 1000 
            
            self.animation.setStartValue(collapsed_height)
            self.animation.setEndValue(target_height)
            self.animation.start()
        else:
            self.is_collapsed = True
            self.toggle_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarUnshadeButton))
            
            start_height = self.content_frame.height()
            
            self.animation.setStartValue(start_height)
            self.animation.setEndValue(0) # Collapse to 0
            
            self.animation.finished.connect(lambda: self.content_frame.setVisible(False), 
                                           Qt.ConnectionType.SingleShotConnection)
            
            self.animation.start()

class SynchronizedGraphicsView(QGraphicsView):
    def __init__(self, scene, name="View", parent=None):
        super().__init__(scene, parent)
        self.name = name
        
        self.setMouseTracking(True) 
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.sibling = None
        self.panning = False
        self.pan_start = QPointF()
        self.controller = None
        
        self.is_painting = False
        self.last_paint_pos = None
        self.box_start = None
        self.temp_box_item = None
        
        self.brush_cursor_item = QGraphicsEllipseItem(0, 0, 10, 10)
        self.brush_cursor_item.setPen(QPen(QColor(0, 0, 0, 255), 1)) 
        self.brush_cursor_outer = QGraphicsEllipseItem(0, 0, 10, 10, self.brush_cursor_item)
        self.brush_cursor_outer.setPen(QPen(QColor(255, 255, 255, 255), 1))
        
        self.brush_cursor_item.setBrush(QBrush(Qt.BrushStyle.NoBrush)) 
        
        self.brush_cursor_item.setZValue(99999) 
        
        self.scene().addItem(self.brush_cursor_item)
        self.brush_cursor_item.hide()

    def dragEnterEvent(self, event):
        # Pass the event to the parent to handle
        event.ignore()

    def dropEvent(self, event):
        # Pass the event to the parent to handle
        event.ignore()

    def set_sibling(self, sibling_view):
        self.sibling = sibling_view
        self.horizontalScrollBar().valueChanged.connect(self.sync_scroll_h)
        self.verticalScrollBar().valueChanged.connect(self.sync_scroll_v)

    def set_controller(self, ctrl):
        self.controller = ctrl

    def sync_scroll_h(self, value):
        if self.sibling and self.sibling.horizontalScrollBar().value() != value:
            self.sibling.horizontalScrollBar().setValue(value)

    def sync_scroll_v(self, value):
        if self.sibling and self.sibling.verticalScrollBar().value() != value:
            self.sibling.verticalScrollBar().setValue(value)

    def update_point_scales(self):
        # Get the X scaling factor (m11) from the transform matrix
        zoom = self.transform().m11()
        
        if zoom == 0: zoom = 1
        
        # If zoom is 2.0 (200%), scale should be 0.5 to look normal
        inverse_scale = 1.0 / zoom
        
        for item in self.scene().items():
            # Check for the tag we set in handle_sam_point
            if item.data(0) == "sam_point":
                item.setScale(inverse_scale)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        
        is_touchpad_gesture = (event.phase() != Qt.ScrollPhase.NoScrollPhase) or not pixel_delta.isNull()
        
        if is_touchpad_gesture and not is_ctrl:
            # --- PANNING ---
            
            # Calculate movement vector
            if not pixel_delta.isNull():
                # Use exact pixels from trackpad
                dx = pixel_delta.x()
                dy = pixel_delta.y()
            else:
                # Fallback: Convert angle notches to pixels
                # Standard mouse wheel step is 120. We scale it down for panning.
                dx = angle_delta.x() / 8
                dy = angle_delta.y() / 8

            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            
            # Subtract delta to pan naturally
            hs.setValue(int(hs.value() - dx))
            vs.setValue(int(vs.value() - dy))
            
            # Sync Sibling
            if self.sibling:
                self.sibling.horizontalScrollBar().setValue(hs.value())
                self.sibling.verticalScrollBar().setValue(vs.value())
            
            event.accept()
            return

        # --- ZOOMING ---
        
        target_viewport_pos = event.position().toPoint()
        target_scene_pos = self.mapToScene(target_viewport_pos)
        
        delta = angle_delta.y()
        if delta == 0: return
        
        zoom_in = delta > 0
        factor = DEFAULT_ZOOM_FACTOR if zoom_in else 1 / DEFAULT_ZOOM_FACTOR
        
        current_scale = self.transform().m11()
        
        view_rect = self.viewport().rect()
        scene_rect = self.sceneRect()
        
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            ratio_w = view_rect.width() / scene_rect.width()
            ratio_h = view_rect.height() / scene_rect.height()
            min_scale = min(ratio_w, ratio_h)
        else:
            min_scale = 0.01 
            
        max_scale = 10.0 
        
        new_scale = current_scale * factor
        
        if new_scale > max_scale:
            factor = max_scale / current_scale
            if factor <= 1.0 and zoom_in: return 
        elif new_scale < min_scale:
            factor = min_scale / current_scale
            if factor >= 1.0 and not zoom_in: return 
        
        self.scale(factor, factor)
        self.update_point_scales()
        
        # Recenter zoom on mouse
        new_viewport_pos = self.mapFromScene(target_scene_pos)
        delta_viewport = new_viewport_pos - target_viewport_pos
        
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta_viewport.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta_viewport.y())
        
        if self.sibling:
            self.sibling.setTransform(self.transform())
            self.sibling.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
            self.sibling.verticalScrollBar().setValue(self.verticalScrollBar().value())
            self.sibling.update_point_scales()
            
        if self.controller:
            self.controller.update_zoom_label()
            self.update_brush_cursor(target_scene_pos)
            if self.sibling:
                self.sibling.update_brush_cursor(target_scene_pos)
        
        event.accept()

    def keyPressEvent(self, event):
        pan_step = 20 # Pixels to move
        if event.key() == Qt.Key.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - pan_step)
        elif event.key() == Qt.Key.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + pan_step)
        elif event.key() == Qt.Key.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - pan_step)
        elif event.key() == Qt.Key.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + pan_step)
        else:
            super().keyPressEvent(event)

    def update_brush_cursor(self, scene_pos):
        if not self.controller or not self.controller.paint_mode:
            if self.brush_cursor_item.isVisible():
                self.brush_cursor_item.hide()
            return
        
        if not self.brush_cursor_item.isVisible():
            self.brush_cursor_item.show()
            
        zoom = self.transform().m11()
        if zoom == 0: zoom = 1
        scene_dia = PAINT_BRUSH_SCREEN_SIZE / zoom
        r = scene_dia / 2
        
        rect_centered = QRectF(-r, -r, scene_dia, scene_dia)
        self.brush_cursor_item.setRect(rect_centered)
        self.brush_cursor_outer.setRect(rect_centered)
        self.brush_cursor_item.setPos(scene_pos)
        
        pw = max(1.0, 1.0 / zoom)
        self.brush_cursor_item.setPen(QPen(QColor(0,0,0), pw))
        self.brush_cursor_outer.setPen(QPen(QColor(255,255,255), pw, Qt.PenStyle.DashLine))

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self.controller:
            if self.controller.paint_mode:
                if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
                    self.is_painting = True
                    self.last_paint_pos = scene_pos
                    self.controller.handle_paint_start(scene_pos)
                    event.accept()
                    return
            else:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.box_start = scene_pos
                    self.temp_box_item = QGraphicsRectItem()
                    zoom = self.transform().m11()
                    pen_width = 2.0 / zoom if zoom != 0 else 2.0
                    self.temp_box_item.setPen(QPen(Qt.GlobalColor.red, pen_width))
                    self.scene().addItem(self.temp_box_item)
                    self.temp_box_item.hide()
                    event.accept()
                elif event.button() == Qt.MouseButton.RightButton:
                    self.controller.handle_sam_point(scene_pos, is_positive=False)
                    event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        
        self.update_brush_cursor(scene_pos)
        if self.sibling:
            self.sibling.update_brush_cursor(scene_pos)

        if self.panning:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            hs.setValue(hs.value() - int(delta.x()))
            vs.setValue(vs.value() - int(delta.y()))
            event.accept()
            return
            
        if self.controller and self.controller.paint_mode and self.is_painting:
            self.controller.handle_paint_move(self.last_paint_pos, scene_pos)
            self.last_paint_pos = scene_pos
            event.accept()
            return
            
        if self.box_start and self.temp_box_item:
            rect = QRectF(self.box_start, scene_pos).normalized()
            self.temp_box_item.setRect(rect)
            zoom = self.transform().m11()
            if zoom == 0: zoom = 1
            min_size_in_scene = MIN_RECT_SIZE / zoom

            if rect.width() >= min_size_in_scene or rect.height() >= min_size_in_scene:
                self.temp_box_item.show()
                for item in self.scene().items():
                    if item.data(0) == "sam_point":
                        self.scene().removeItem(item)
            else:
                self.temp_box_item.hide()
            event.accept()
            return
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Handle Panning Release
        if self.panning and event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        # Handle Paint Release (Left OR Right button)
        if self.controller and self.controller.paint_mode:
            if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
                self.is_painting = False
                self.last_paint_pos = None
                self.controller.handle_paint_end()
                
        # Handle Box Selection Release (Left Only)
        if self.box_start and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.box_start, scene_pos).normalized()
            if self.temp_box_item:
                self.scene().removeItem(self.temp_box_item)
                self.temp_box_item = None
            self.box_start = None
            
            # Convert MIN_RECT_SIZE from screen pixels to scene pixels
            zoom = self.transform().m11()
            if zoom == 0: zoom = 1
            min_size_in_scene = MIN_RECT_SIZE / zoom

            # Logic to distinguish between a Click (Point) and a Drag (Box)
            if rect.width() < min_size_in_scene and rect.height() < min_size_in_scene:
                if self.controller: 
                    # If we are in paint mode, clicks are paint dots, not SAM points
                    if not self.controller.paint_mode:
                        self.controller.handle_sam_point(scene_pos, is_positive=True)
            else:
                if self.controller and not self.controller.paint_mode: 
                    self.controller.handle_sam_box(rect)
                    
        super().mouseReleaseEvent(event)
        
    def leaveEvent(self, event):
        # --- Clean hide on both screens when leaving ---
        if self.controller and self.controller.paint_mode:
            self.brush_cursor_item.hide()
            if self.sibling:
                # Direct access to sibling item to ensure sync
                self.sibling.brush_cursor_item.hide()
        super().leaveEvent(event)

class ImageEditorDialog(QDialog):
    def __init__(self, parent, image):
        super().__init__(parent)
        self.setWindowTitle("Preprocess and Crop Image")
        self.resize(1200, 800)
        
        # --- Data Setup ---
        self.original_image = image
        self.final_image = None
        self.total_rotation = 0
        
        # Create a small preview for real-time editing (Max 1000px dimension)
        w, h = image.size
        scale = min(1.0, 1000.0 / max(w, h))
        if scale < 1.0:
            self.preview_base = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        else:
            self.preview_base = image.copy()
            
        # The current state of the preview
        self.display_image = self.preview_base
        
        # --- UI Setup ---
        self.crop_start = None
        self.crop_rect_item = None
        
        layout = QHBoxLayout(self)
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view, stretch=2)
        
        self.view.viewport().installEventFilter(self)
        
        # --- Controls ---
        controls_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        self.sliders_layout = QVBoxLayout(scroll_widget)
        
        self.sliders = {}
        self.slider_params = {
            'highlight': (0.1, 2.0, 1.0), 'midtone': (0.1, 2.0, 1.0), 'shadow': (0.1, 3.0, 1.0),
            'tone_curve': (0.01, 0.5, 0.1), 'brightness': (0.1, 2.0, 1.0), 'contrast': (0.1, 2.0, 1.0),
            'saturation': (0.1, 2.0, 1.0), 'white_balance': (2000, 10000, 6500), 'unsharp_radius': (0.1, 50, 1.0),
            'unsharp_amount': (0, 500, 0), 'unsharp_threshold': (0, 255, 0)
        }
        
        # Update timer to prevent lag while dragging
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(50) # 50ms 
        self.update_timer.timeout.connect(self.update_preview)

        for param, (min_v, max_v, default) in self.slider_params.items():
            lbl = QLabel(param.replace("_", " ").capitalize())
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            val_norm = (default - min_v) / (max_v - min_v)
            slider.setValue(int(val_norm * 100))
            
            # Connect to timer instead of function directly
            slider.valueChanged.connect(self.update_timer.start)
            
            self.sliders_layout.addWidget(lbl)
            self.sliders_layout.addWidget(slider)
            self.sliders[param] = {'widget': slider, 'min': min_v, 'max': max_v}
            
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        controls_layout.addWidget(scroll)

        rot_layout = QHBoxLayout()
        btn_rot_l = QPushButton("Rotate Left"); btn_rot_l.clicked.connect(lambda: self.rotate_image(90))
        btn_rot_r = QPushButton("Rotate Right"); btn_rot_r.clicked.connect(lambda: self.rotate_image(-90))
        rot_layout.addWidget(btn_rot_l); rot_layout.addWidget(btn_rot_r)
        controls_layout.addLayout(rot_layout)
        
        btn_reset = QPushButton("Reset to Original"); btn_reset.clicked.connect(self.reset_sliders)
        controls_layout.addWidget(btn_reset)
        
        crop_btn = QPushButton("Apply to Full Image & Save"); crop_btn.clicked.connect(self.apply_full_res_and_accept)
        controls_layout.addWidget(crop_btn)
        layout.addLayout(controls_layout, stretch=1)
        
        self.update_preview()

    def showEvent(self, event):
        super().showEvent(event)
        if self.pixmap_item.pixmap():
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def eventFilter(self, source, event):
        if source == self.view.viewport():
            if event.type() == event.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self.crop_start = self.view.mapToScene(event.pos())
                if self.crop_rect_item: self.scene.removeItem(self.crop_rect_item)
                self.crop_rect_item = QGraphicsRectItem()
                self.crop_rect_item.setPen(QPen(Qt.GlobalColor.red, 2))
                self.scene.addItem(self.crop_rect_item)
                return True
            elif event.type() == event.Type.MouseMove and self.crop_start:
                curr = self.view.mapToScene(event.pos())
                rect = QRectF(self.crop_start, curr).normalized()
                self.crop_rect_item.setRect(rect)
                return True
            elif event.type() == event.Type.MouseButtonRelease and self.crop_start:
                self.crop_start = None
                return True
        return super().eventFilter(source, event)

    def get_val(self, param):
        s = self.sliders[param]
        return s['min'] + (s['widget'].value() / 100.0 * (s['max'] - s['min']))
    
    def reset_sliders(self):
        self.update_timer.stop()
        for param, (min_v, max_v, default) in self.slider_params.items():
            val_norm = (default - min_v) / (max_v - min_v)
            self.sliders[param]['widget'].setValue(int(val_norm * 100))
        self.total_rotation = 0
        self.update_preview()

    def rotate_image(self, angle):
        self.total_rotation = (self.total_rotation + angle) % 360
        # We don't rotate the PIL images here to avoid quality loss. 
        # We just store the rotation and apply it during the pipeline.
        self.update_preview()

    def process_image_pipeline(self, input_img):
        """Applies all current filters to the given input_img"""
        
        # 1. Rotate
        if self.total_rotation != 0:
            img = input_img.rotate(self.total_rotation, expand=True)
        else:
            img = input_img
            
        img_array = np.array(img)
        
        # 2. White Balance
        wb = self.get_val('white_balance')
        if wb != 6500: img_array = self.adjust_white_balance(img_array, wb)
        
        # 3. Tone Curves (LUT)
        x = np.arange(256, dtype=np.float32)
        tone = self.get_val('tone_curve')
        hl_mask = sigmoid((x - 192) / (255 * tone))
        sh_mask = 1 - sigmoid((x - 64) / (255 * tone))
        mt_mask = 1 - hl_mask - sh_mask
        lut = (x * self.get_val('highlight') * hl_mask + x * self.get_val('midtone') * mt_mask + x * self.get_val('shadow') * sh_mask).clip(0, 255).astype(np.uint8)
        
        if len(img_array.shape) == 3:
            # If RGBA, only apply to RGB
            if img_array.shape[2] >= 3:
                img_array[:,:,:3] = lut[img_array[:,:,:3]]
        else: 
            img_array = lut[img_array]
        
        img = Image.fromarray(img_array)
        
        # Pillow Enhancements
        if self.get_val('brightness') != 1.0: img = ImageEnhance.Brightness(img).enhance(self.get_val('brightness'))
        if self.get_val('contrast') != 1.0: img = ImageEnhance.Contrast(img).enhance(self.get_val('contrast'))
        if self.get_val('saturation') != 1.0: img = ImageEnhance.Color(img).enhance(self.get_val('saturation'))
        
        # Unsharp Mask
        amt = self.get_val('unsharp_amount')
        if amt > 0: 
            img = img.filter(ImageFilter.UnsharpMask(
                radius=int(self.get_val('unsharp_radius')), 
                percent=int(amt), 
                threshold=int(self.get_val('unsharp_threshold'))
            ))
            
        return img

    def update_preview(self):
        self.display_image = self.process_image_pipeline(self.preview_base)
        
        # Update View
        self.pixmap_item.setPixmap(pil2pixmap(self.display_image))
        self.view.setSceneRect(self.pixmap_item.boundingRect())
        # Only fit in view if it's the first load or rotation changed significantly
        if not self.view.sceneRect().isEmpty():
             self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def adjust_white_balance(self, img_array, temp):
        t = temp / 100
        if t <= 66:
            r, g, b = 255, 99.47 * math.log(t) - 161.1, (0 if t<=19 else 138.5 * math.log(t-10) - 305)
        else:
            r, g, b = 329.7 * math.pow(t-60, -0.133), 288.1 * math.pow(t-60, -0.075), 255
        
        r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        
        # Using numpy vectorized operations for speed
        if img_array.shape[2] >= 3:
            rgb = img_array[:,:,:3].astype(float)
            avg = np.mean(rgb)
            
            # Avoid division by zero
            r_scale = 255/max(1, r)
            g_scale = 255/max(1, g)
            b_scale = 255/max(1, b)
            
            rgb[:,:,0] *= r_scale
            rgb[:,:,1] *= g_scale
            rgb[:,:,2] *= b_scale
            
            current_avg = np.mean(rgb)
            if current_avg > 0:
                rgb *= (avg / current_avg)
                
            img_array[:,:,:3] = np.clip(rgb, 0, 255).astype(np.uint8)
            
        return img_array

    def apply_full_res_and_accept(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            full_res_processed = self.process_image_pipeline(self.original_image)
            
            if self.crop_rect_item:
                rect = self.crop_rect_item.rect().normalized()
                
                # We need to map the crop rectangle from the Preview scale to the Full Res scale
                preview_w = self.pixmap_item.pixmap().width()
                preview_h = self.pixmap_item.pixmap().height()
                full_w = full_res_processed.width
                full_h = full_res_processed.height
                
                scale_x = full_w / preview_w
                scale_y = full_h / preview_h
                
                x = int(rect.x() * scale_x)
                y = int(rect.y() * scale_y)
                w = int(rect.width() * scale_x)
                h = int(rect.height() * scale_y)
                
                if w > 0 and h > 0:
                    self.final_image = full_res_processed.crop((x, y, x+w, y+h))
                else:
                    self.final_image = full_res_processed
            else: 
                self.final_image = full_res_processed
                
            self.accept()
        finally:
            QApplication.restoreOverrideCursor()

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
        self.loaded_whole_models = {}
        self.loaded_sam_models = {} 
        self.model_output_mask = None

        self.init_ui()
        self.setup_keybindings()

        self.shadow_timer = QTimer()
        self.shadow_timer.setSingleShot(True)
        self.shadow_timer.setInterval(50) # Wait 50ms after last movement
        self.shadow_timer.timeout.connect(self.update_output_preview)

        # Let the UI build before loading image, so the correct display zoom is shown
        # 30ms seems long enough on my PC
        if self.image_paths:
            QTimer.singleShot(30, lambda: self.load_image(self.image_paths[0]))
        else:
            self.load_blank_image()
        
        # Delay until UI created
        QTimer.singleShot(10, self.update_cached_model_icons)
        # Delay model pre-loading
        QTimer.singleShot(100, self.preload_startup_models)

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
        btn_o = QPushButton("Open"); btn_o.clicked.connect(self.load_image_dialog)
        btn_c = QPushButton("Clipboard"); btn_c.clicked.connect(self.load_clipboard)
        self.btn_next = QPushButton("Next >")
        self.btn_next.clicked.connect(self.load_next_image)
        nav.addWidget(btn_o); nav.addWidget(btn_c); nav.addWidget(self.btn_next)
        sl.addLayout(nav)
        
        btn_ed = QPushButton("Edit Image"); btn_ed.clicked.connect(self.open_image_editor)
        btn_lm = QPushButton("Load Mask"); btn_lm.clicked.connect(self.load_mask_dialog)

        h_edit_load_buttons = QHBoxLayout()
        h_edit_load_buttons.addWidget(btn_ed)
        h_edit_load_buttons.addWidget(btn_lm)
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
        labelS.setContentsMargins(3, 0, 0, 0)   # push right by 10px
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
        self.combo_bg = QComboBox()
        
        colors = ["Transparent", "White", "Black", "Red", "Blue", 
                  "Orange", "Yellow", "Green", "Grey", 
                  "Lightgrey", "Brown", "Blurred (Slow)"]
        self.combo_bg.addItems(colors)
        self.combo_bg.currentTextChanged.connect(self.handle_bg_change)
        sl.addWidget(self.combo_bg)
        
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
        btn_help = QPushButton("Help / About"); btn_help.clicked.connect(self.show_help)
        sl.addWidget(btn_help)

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
        self.view_input = SynchronizedGraphicsView(self.scene_input, name="Input View") 
        self.view_input.set_controller(self)
        self.view_input.setBackgroundBrush(QBrush(hatch_color, Qt.BrushStyle.DiagCrossPattern))
        self.view_input.setAcceptDrops(True)


        self.input_pixmap_item = QGraphicsPixmapItem(); self.scene_input.addItem(self.input_pixmap_item)
        self.overlay_pixmap_item = QGraphicsPixmapItem(); self.overlay_pixmap_item.setOpacity(0.5); self.scene_input.addItem(self.overlay_pixmap_item)
        
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
        self.view_output = SynchronizedGraphicsView(self.scene_output, name="Output View")
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
        
        is_dark_mode = QApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark

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
        QShortcut(QKeySequence("Ctrl+Shift+S"), self).activated.connect(self.quick_save_jpeg)
        
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
            self.status_label.setText(f"Loaded: {os.path.basename(path)}")
            self.update_next_button_state()
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
            self.status_label.setText("Loaded from Clipboard")
            self.update_window_title()
            if hasattr(self, 'update_next_button_state'):
                self.update_next_button_state()
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
                self.update_next_button_state()

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

    def update_next_button_state(self):
        if not self.image_paths:
            self.btn_next.setEnabled(False)
            return

        is_clipboard = (self.image_paths[0] == "Clipboard")
        has_more_images = (self.current_image_index < len(self.image_paths) - 1)
        
        self.btn_next.setEnabled(has_more_images and not is_clipboard)

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
                session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
                load_time = (timer() - t_start) * 1000
                self.update_cached_model_icons()

                if cache_mode == 1: # Cache Last
                    # Clear previous models before adding the new one
                    if self.loaded_whole_models:
                        for key, sess in self.loaded_whole_models.items():
                            del sess
                        self.loaded_whole_models.clear()
                        gc.collect()
                    self.loaded_whole_models[cache_key] = session
                elif cache_mode == 2: # Cache All
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

    def clear_overlay(self):
        self.coordinates = []
        self.labels = []
        self.model_output_mask = Image.new("L", self.original_image.size, 0)
        self.overlay_pixmap_item.setPixmap(QPixmap())
        
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
    
    def modify_mask(self, op):
        if not self.model_output_mask: return
        self.add_undo_step()
        m = self.model_output_mask
        
        if self.chk_soften.isChecked():
            m = m.filter(ImageFilter.GaussianBlur(radius=SOFTEN_RADIUS))
            
        if self.chk_post.isChecked():
            # Convert to boolean array for morphological operations
            arr = np.array(m) > 128
            kernel = np.ones((3,3), np.uint8)
            arr = cv2.erode(cv2.dilate(arr.astype(np.uint8), kernel, iterations=1), kernel, iterations=1).astype(bool)
            m = Image.fromarray((arr*255).astype(np.uint8))
        self.working_mask = op(self.working_mask, m)
        self.update_output_preview()

    def toggle_shadow_options(self, checked):
        if checked: self.shadow_frame.show()
        else: self.shadow_frame.hide()
        self.update_output_preview()

    def render_output_image(self, shadow_downscale=0.125):
        if not self.original_image: return
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

        bg_txt = self.combo_bg.currentText()
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

    def update_output_preview(self, *args):

        if self.chk_show_mask.isChecked(): 
            final = self.working_mask.convert("RGBA")
        else:
            final = self.render_output_image()
        
        self.output_pixmap_item.setPixmap(pil2pixmap(final))
        self.view_output.setSceneRect(self.view_input.sceneRect())

    def reset_all(self):
        self.clear_overlay()
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
    # force set app dark scheme on linux, seems to ignore system scheme
    # app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    window = BackgroundRemoverGUI(sys.argv[1:])
    window.showMaximized()
    sys.exit(app.exec())