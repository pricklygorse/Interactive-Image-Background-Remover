from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QDialog, QScrollArea, QComboBox,
                             QGraphicsRectItem, QRadioButton, QButtonGroup, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPoint
from PyQt6.QtGui import (QColor, QPen, QBrush, QImage, QPainter, QPainterPath, QShortcut, QKeySequence, QCursor)

from PIL import Image, ImageFilter
import math
import numpy as np

from timeit import default_timer as timer
import cv2


from src.utils import *



class InpaintingView(QGraphicsView):
    """
    Custom View for the Inpainting Dialog. 
    Handles drawing (Left Click) and erasing (Right Click).
    """
    def __init__(self, scene, dialog):
        super().__init__(scene)
        self.dialog = dialog
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setMouseTracking(True)
        
        self._panning = False
        self._pan_start = QPoint()

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.15 if zoom_in else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton]:
            self.dialog.drawing = True
            self.dialog.is_erasing = (event.button() == Qt.MouseButton.RightButton)
            scene_pos = self.mapToScene(event.pos())
            self.dialog.start_stroke(scene_pos)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start = event.pos()
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())
        self.dialog.update_brush_cursor(scene_pos)
        
        if self.dialog.drawing:
            self.dialog.continue_stroke(scene_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        elif event.button() in [Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton] and self.dialog.drawing:
            self.dialog.drawing = False
            self.dialog.end_stroke()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

class InpaintingDialog(QDialog):
    def __init__(self, image_pil, model_manager, provider_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inpainting")
        self.resize(1300, 850)
        self.setModal(True)

        self.model_manager = model_manager
        self.provider_data = provider_data
        
        # Current State
        self.base_image = image_pil.convert("RGBA")
        self.mask_image = Image.new("L", self.base_image.size, 0)
        
        # Single-step Undo Storage
        self.undo_state = None 

        self.drawing = False
        self.is_erasing = False
        self.brush_size = 40
        self.context_padding = 150 # Pixels around mask
        self.current_path = QPainterPath()
        self.temp_path_item = None
        
        self.init_ui()
        self.update_view_images()
        self.update_context_visual()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Info Header
        info_lbl = QLabel(
            "<b>Inpainting.</b> Works best with small images and objects. Output quality can vary dramatically.<br><br>"
            "<b>LaMa:</b> Newer. Inference time is strongly proportional to image size. Works best with smaller images or objects. Can reduce inference time by lowering the context padding, which may reduce quality.<br>"
            "<b>DeepFillv2:</b> Models trained on celebrity faces or places. Inference time and quality is set based on the model's internal resolution.<br>"
            "<b>Usage:</b> Left click to draw, right click to erase. Specify how much context to send to the model"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setStyleSheet("color: #bbb; padding: 10px; background: #222; border: 1px solid #444; border-radius: 4px;")
        layout.addWidget(info_lbl)

        # Toolbar
        toolbar = QFrame()
        tb_layout = QHBoxLayout(toolbar)
        
        tb_layout.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.setToolTip("Select the inpainting algorithm. Only downloaded models are shown.")
        
        # Check which models present
        available_models = []
        root = self.model_manager.model_root_dir
        
        if os.path.exists(os.path.join(root, "lama.onnx")):
            available_models.append("LaMa")
            
        df_variants = [
            'deepfillv2_celeba_256x256',
            'deepfillv2_places_256x256', 'deepfillv2_places_512x512', 'deepfillv2_places_1024x1024'
        ]
        for variant in df_variants:
            if os.path.exists(os.path.join(root, f"{variant}.onnx")):
                available_models.append(variant)
        
        if not available_models:
            self.combo_model.addItem("No models found")
            self.combo_model.setEnabled(False)
        else:
            self.combo_model.addItems(available_models)
            
        tb_layout.addWidget(self.combo_model)
        
        tb_layout.addSpacing(20)
        

        tb_layout.addWidget(QLabel("Brush Size:"))
        self.sl_size = QSlider(Qt.Orientation.Horizontal)
        self.sl_size.setRange(5, 300)
        self.sl_size.setValue(self.brush_size)
        self.sl_size.setFixedWidth(150)
        self.sl_size.valueChanged.connect(self.set_brush_size)
        tb_layout.addWidget(self.sl_size)

        tb_layout.addSpacing(20)

        tb_layout.addWidget(QLabel("Context Padding:"))
        self.sl_context = QSlider(Qt.Orientation.Horizontal)
        # Set a temporary range; will be corrected by update_context_slider_range
        self.sl_context.setRange(50, 1000) 
        self.sl_context.setValue(self.context_padding)
        self.sl_context.setFixedWidth(150)
        self.sl_context.setToolTip("Sets the size of the area processed around your lines. Smaller = Faster.")
        self.sl_context.valueChanged.connect(self.set_context_padding)
        tb_layout.addWidget(self.sl_context)
        
        # Initialize the correct range based on the image immediately
        self.update_context_slider_range()
        
        tb_layout.addStretch()

        self.btn_undo = QPushButton("Undo (Ctrl+Z)")
        self.btn_undo.clicked.connect(self.undo_last)
        self.btn_undo.setEnabled(False)
        tb_layout.addWidget(self.btn_undo)
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_last)

        self.btn_clear = QPushButton("Clear Strokes (C)")
        self.btn_clear.setToolTip("Remove all drawn mask strokes.")
        self.btn_clear.clicked.connect(self.clear_mask)
        tb_layout.addWidget(self.btn_clear)
        QShortcut(QKeySequence("C"), self).activated.connect(self.clear_mask)

        
        self.btn_run = QPushButton("Run Inpainting")
        self.btn_run.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 8px 20px;")
        self.btn_run.clicked.connect(self.run_inference)
        tb_layout.addWidget(self.btn_run)
        
        layout.addWidget(toolbar)
        
        self.scene = QGraphicsScene()
        self.view = InpaintingView(self.scene, self)
        layout.addWidget(self.view)
        
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_apply = QPushButton("Apply to Main UI")
        btn_apply.setStyleSheet("font-weight: bold;")
        btn_apply.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_apply)
        layout.addLayout(btn_layout)
        
        self.bg_item = QGraphicsPixmapItem()
        self.scene.addItem(self.bg_item)
        
        self.mask_overlay_item = QGraphicsPixmapItem()
        self.mask_overlay_item.setOpacity(0.8) 
        self.scene.addItem(self.mask_overlay_item)

        self.context_rect_item = QGraphicsRectItem()
        context_pen = QPen(QColor(0, 255, 255, 180), 2, Qt.PenStyle.DashLine)
        context_pen.setCosmetic(True)
        self.context_rect_item.setPen(context_pen)
        self.scene.addItem(self.context_rect_item)
        
        self.cursor_item = self.scene.addEllipse(0,0,1,1, QPen(Qt.GlobalColor.white, 1))
        self.cursor_item.setZValue(2000)

    def clear_mask(self):
        """
        Resets the mask to empty and updates all visuals and slider ranges.
        """
        self.mask_image = Image.new("L", self.base_image.size, 0)
        self.update_view_images()
        self.update_context_visual()
        self.update_context_slider_range()

    def update_context_slider_range(self):
        """
        Dynamically updates the slider maximum so it can always reach the image edges.
        """
        bbox = self.mask_image.getbbox()
        img_w, img_h = self.base_image.size
        
        if not bbox:
            # If no mask, allow padding up to the largest image dimension
            max_p = max(img_w, img_h)
        else:
            x1, y1, x2, y2 = bbox
            # Calculate distance to all 4 edges
            dist_left = x1
            dist_top = y1
            dist_right = img_w - x2
            dist_bottom = img_h - y2
            # The maximum of these distances is the padding needed to cover the whole image
            max_p = max(dist_left, dist_top, dist_right, dist_bottom)

        # Ensure a sensible minimum max (e.g., 1000) so the slider isn't too tiny
        final_max = max(1000, max_p)
        
        self.sl_context.blockSignals(True)
        self.sl_context.setMaximum(int(final_max))
        self.sl_context.blockSignals(False)


    def set_brush_size(self, val):
        self.brush_size = val
        self.update_brush_cursor(self.view.mapToScene(self.view.mapFromGlobal(QCursor.pos())))

    def set_context_padding(self, val):
        self.context_padding = val
        self.update_context_visual()
        
    def update_brush_cursor(self, pos):
        r = self.brush_size / 2
        self.cursor_item.setRect(pos.x()-r, pos.y()-r, self.brush_size, self.brush_size)

    def update_view_images(self):
        self.bg_item.setPixmap(pil2pixmap(self.base_image))
        self.scene.setSceneRect(self.bg_item.boundingRect())
        
        # Visualise mask as red overlay
        mask_np = np.array(self.mask_image)
        h, w = mask_np.shape
        red_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        red_overlay[:,:,0] = 255 
        red_overlay[:,:,3] = mask_np 
        self.mask_overlay_item.setPixmap(numpy_to_pixmap(red_overlay))

    def update_context_visual(self):
        bbox = self.mask_image.getbbox()
        if not bbox:
            self.context_rect_item.hide()
            return

        x1, y1, x2, y2 = bbox
        img_w, img_h = self.base_image.size
        
        # Find the center of the current mask
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Determine the side length (maximum dimension of mask + padding)
        # We add padding to both sides, so we use max_dim + (2 * padding)
        max_mask_dim = max(x2 - x1, y2 - y1)
        side = max_mask_dim + (self.context_padding * 2)
        half_side = side / 2
        
        # Calculate square coordinates
        cx1 = mid_x - half_side
        cy1 = mid_y - half_side
        cx2 = mid_x + half_side
        cy2 = mid_y + half_side
        
        # Clamp to image boundaries
        final_cx1 = max(0, cx1)
        final_cy1 = max(0, cy1)
        final_cx2 = min(img_w, cx2)
        final_cy2 = min(img_h, cy2)
        
        self.context_rect_item.setRect(QRectF(final_cx1, final_cy1, final_cx2 - final_cx1, final_cy2 - final_cy1))
        self.context_rect_item.show()

    def start_stroke(self, pos):
        self.current_path = QPainterPath(pos)
        color = QColor(255, 0, 0, 200) if not self.is_erasing else QColor(0, 0, 0, 200)
        pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.temp_path_item = self.scene.addPath(self.current_path, pen)

    def continue_stroke(self, pos):
        self.current_path.lineTo(pos)
        if self.temp_path_item:
            self.temp_path_item.setPath(self.current_path)

    def end_stroke(self):
        if self.temp_path_item:
            self.scene.removeItem(self.temp_path_item)
            self.temp_path_item = None
            
        qimg = QImage(self.mask_image.width, self.mask_image.height, QImage.Format.Format_Grayscale8)
        qimg.fill(0)
        p = QPainter(qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False) 
        p.setPen(QPen(QColor(255), self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        p.drawPath(self.current_path)
        p.end()
        
        ptr = qimg.constBits()
        ptr.setsize(qimg.sizeInBytes())
        stroke_np = np.array(ptr, copy=False).reshape(qimg.height(), qimg.bytesPerLine())[:, :qimg.width()]
        
        if self.is_erasing:
            mask_np = np.array(self.mask_image)
            mask_np[stroke_np > 0] = 0
            self.mask_image = Image.fromarray(mask_np)
        else:
            self.mask_image.paste(255, (0,0), Image.fromarray(stroke_np))
        
        self.update_view_images()
        self.update_context_visual()
        # Update slider range because the mask bounding box has changed
        self.update_context_slider_range()

    def undo_last(self):
        if not self.undo_state:
            return
            
        self.base_image = self.undo_state["image"].copy()
        self.mask_image = self.undo_state["mask"].copy()
        
        self.update_view_images()
        self.update_context_visual()
        
        self.update_context_slider_range()
        
        self.undo_state = None
        self.btn_undo.setEnabled(False)

    def run_inference(self):
        bbox = self.mask_image.getbbox()
        if not bbox:
            return

        self.undo_state = {
            "image": self.base_image.copy(),
            "mask": self.mask_image.copy()
        }
        self.btn_undo.setEnabled(True)

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            img_w, img_h = self.base_image.size
            x1, y1, x2, y2 = bbox
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            max_mask_dim = max(x2 - x1, y2 - y1)
            side = max_mask_dim + (self.context_padding * 2)
            half_side = side / 2

            cx1 = int(max(0, mid_x - half_side))
            cy1 = int(max(0, mid_y - half_side))
            cx2 = int(min(img_w, mid_x + half_side))
            cy2 = int(min(img_h, mid_y + half_side))
            
            image_patch = self.base_image.crop((cx1, cy1, cx2, cy2))
            mask_patch = self.mask_image.crop((cx1, cy1, cx2, cy2))

            algo = self.combo_model.currentText()
            
            if "deepfill" in algo.lower():
                inpainted_patch, inf_time = self.model_manager.run_deepfill_inpainting(
                    image_patch, mask_patch, self.provider_data, model_name=algo
                )
            else:
                inpainted_patch, inf_time = self.model_manager.run_lama_inpainting(
                    image_patch, mask_patch, self.provider_data
                )
            
            # Paste back with feather
            binary_mask = mask_patch.point(lambda p: 255 if p > 0 else 0)
            soft_mask = binary_mask.filter(ImageFilter.GaussianBlur(radius=1.5))
            
            self.base_image.paste(inpainted_patch, (cx1, cy1), soft_mask)
            self.mask_image = Image.new("L", self.base_image.size, 0)
            
            self.update_view_images()
            self.update_context_visual()
            self.update_context_slider_range()
            
        except Exception as e:
            QMessageBox.critical(self, "Inpainting Error", str(e))
            self.undo_last()
        finally:
            QApplication.restoreOverrideCursor()

    def get_result(self):
        return self.base_image


# not used now changed to tabbed UI
# keeping for now incase required
class SaveOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Options")
        self.setFixedSize(350, 300)
        
        layout = QVBoxLayout(self)
        
        # Format Selection
        grp = QFrame(); v_grp = QVBoxLayout(grp)
        v_grp.addWidget(QLabel("<b>File Format:</b>"))
        
        self.rb_png = QRadioButton("PNG (Lossless)")
        self.rb_webp_lossless = QRadioButton("WebP (Lossless)")
        self.rb_webp_lossy = QRadioButton("WebP (Lossy)")
        self.rb_jpg = QRadioButton("JPEG (No Transparency)")
        
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
        btn_ok = QPushButton("OK"); btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel"); btn_cancel.clicked.connect(self.reject)
        btns.addWidget(btn_ok); btns.addWidget(btn_cancel)
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
    


# not used anymore but keeping just incase
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
        # w, h = image.size
        # scale = min(1.0, 1000.0 / max(w, h))
        # if scale < 1.0:
        #     self.preview_base = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
        # else:
        #     self.preview_base = image.copy()
        
        # dont need smaller preview any more now it is so optimised
        self.preview_base = image.copy()
        self.preview_base_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.preview_base.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
        )
            
        # The current state of the preview
        self.display_image = self.preview_base
        self.display_image_np = self.preview_base_np.copy()
        
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
            # (Min, Max, Default)
            # uses integers for simplicity, converted to float as required later
            'highlight': (10, 200, 100), 
            'midtone': (10, 200, 100), 
            'shadow': (10, 300, 100),
            'tone_curve': (1, 50, 10),     
            'brightness': (10, 200, 100), 
            'contrast': (10, 200, 100),
            'saturation': (0, 200, 100), # Min 0 for grayscale
            'white_balance': (2000, 10000, 6500), 
            'unsharp_radius': (1, 100, 1), 
            'unsharp_amount': (0, 500, 0), 
            'unsharp_threshold': (0, 255, 0)
        }
        
        # Update timer to prevent lag while dragging
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(20) 
        self.update_timer.timeout.connect(self.update_preview)

        for param, (min_v, max_v, default) in self.slider_params.items():
            # Main label for the parameter name
            lbl = QLabel(param.replace("_", " ").capitalize())
            self.sliders_layout.addWidget(lbl)

            # Horizontal layout to hold slider + value label
            h_row_layout = QHBoxLayout()
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default)
            
            # Value label to show the current integer
            val_display = QLabel(str(default))
            val_display.setFixedWidth(35) # Prevents layout jumping when digits change
            val_display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            # Update label and start processing timer on change
            slider.valueChanged.connect(lambda v, l=val_display: l.setText(str(v)))
            slider.valueChanged.connect(lambda _: self.update_timer.start())
            
            h_row_layout.addWidget(slider)
            h_row_layout.addWidget(val_display)
            
            self.sliders_layout.addLayout(h_row_layout)
            self.sliders[param] = slider
            
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
        
        crop_btn = QPushButton("Apply to Image"); crop_btn.clicked.connect(self.apply_full_res_and_accept)
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

   
    def reset_sliders(self):
        self.update_timer.stop()
        for param, (_, _, default) in self.slider_params.items():
            self.sliders[param].setValue(default)
        
        self.total_rotation = 0
        # Re-fetch the clean base numpy array
        self.preview_base_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.preview_base.convert("RGBA")), cv2.COLOR_RGBA2BGRA)
        )
        self.update_preview()

    def rotate_image(self, angle):
        self.total_rotation = (self.total_rotation + angle) % 360
        
        if angle == 90:
            self.preview_base_np = cv2.rotate(self.preview_base_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == -90:
            self.preview_base_np = cv2.rotate(self.preview_base_np, cv2.ROTATE_90_CLOCKWISE)

        self.update_preview()

    def get_current_params(self):
        return {param: slider.value() for param, slider in self.sliders.items()}
    
    

    def update_preview(self):
        params = self.get_current_params()
        self.display_image_np = apply_tone_sharpness(self.preview_base_np, params)
        
        q_img = numpy_bgra_to_pixmap(self.display_image_np)
        
        # Update View
        self.pixmap_item.setPixmap(q_img)
        self.view.setSceneRect(self.pixmap_item.boundingRect())
        
        if not self.view.sceneRect().isEmpty():
            self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    
    def apply_full_res_and_accept(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            full_res_bgra = cv2.cvtColor(np.array(self.original_image.convert("RGBA")), cv2.COLOR_RGBA2BGRA)

            if self.total_rotation == 90:
                full_res_bgra = cv2.rotate(full_res_bgra, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.total_rotation == 180:
                full_res_bgra = cv2.rotate(full_res_bgra, cv2.ROTATE_180)
            elif self.total_rotation == 270 or self.total_rotation == -90:
                full_res_bgra = cv2.rotate(full_res_bgra, cv2.ROTATE_90_CLOCKWISE)

            params = self.get_current_params()
            processed_np = apply_tone_sharpness(full_res_bgra, params)
            
            full_res_processed = Image.fromarray(cv2.cvtColor(processed_np, cv2.COLOR_BGRA2RGBA))
            
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
