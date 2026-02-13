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
            "<b>Inpainting.</b> All these models work best with small images with small objects. Most are trained on 256x256 or 512x512 pixel images, so the output quality can vary dramatically. Use the context window slider to specify what the model sees.<br><br>"
            "<b>LaMa:</b> Resolution independent but very slow at large resolutions. Regardless of resolution independance, it still works best with smaller images with small objects.<br>"
            "<b>MI-GAN:</b> Fast model designed for mobile use. 512x512px output."
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
            
        for filename in os.listdir(root):
            if filename.startswith("migan") and filename.endswith(".onnx"):
                available_models.append(filename.replace(".onnx", ""))
                
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

        # Context Container (Vertical to keep Label above Slider)
        ctx_container = QWidget()
        ctx_vbox = QVBoxLayout(ctx_container)
        ctx_vbox.setContentsMargins(0, 0, 0, 0)
        ctx_vbox.setSpacing(2)

        # Dimension Label
        self.lbl_context_dims = QLabel("Context: 0 x 0")
        self.lbl_context_dims.setStyleSheet("font-size: 14px; color: #aaa; font-weight: bold;")
        self.lbl_context_dims.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctx_vbox.addWidget(self.lbl_context_dims)

        self.sl_context = QSlider(Qt.Orientation.Horizontal)
        self.sl_context.setRange(50, 1000) 
        self.sl_context.setValue(self.context_padding)
        self.sl_context.setFixedWidth(150)
        self.sl_context.setToolTip("Sets the size of the area processed around your lines. Smaller = Faster.")
        self.sl_context.valueChanged.connect(self.set_context_padding)
        ctx_vbox.addWidget(self.sl_context)
        
        tb_layout.addWidget(ctx_container)
        
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
            self.lbl_context_dims.setText("Context: 0 x 0") # Reset label
            return

        x1, y1, x2, y2 = bbox
        img_w, img_h = self.base_image.size
        
        # Find the center of the current mask
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Determine the side length (maximum dimension of mask + padding)
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
        
        # Calculate actual resulting dimensions
        ctx_w = int(final_cx2 - final_cx1)
        ctx_h = int(final_cy2 - final_cy1)
        
        # Update the visual rect
        self.context_rect_item.setRect(QRectF(final_cx1, final_cy1, ctx_w, ctx_h))
        self.context_rect_item.show()

        # Update the Dimension Label
        # We use a color warning if it's significantly larger than 512
        color = "#aaa"
        if ctx_w > 1024 or ctx_h > 1024:
            color = "#ff5555" # Red for very large
        elif ctx_w > 640 or ctx_h > 640:
            color = "#ffaa00" # Orange for slightly large
            
        self.lbl_context_dims.setText(f"Context: {ctx_w} x {ctx_h}")
        self.lbl_context_dims.setStyleSheet(f"font-size: 14px; color: {color}; font-weight: bold;")

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
            elif "migan" in algo.lower():
                inpainted_patch, inf_time = self.model_manager.run_migan_inpainting(
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

