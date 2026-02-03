import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QSlider, QLabel, QFrame, QRadioButton, 
                             QButtonGroup, QSizePolicy, QFileDialog, QMessageBox)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QPainterPath, QPen, QColor, QBrush, QImage

from .utils import pil2pixmap, numpy_to_pixmap

class TrimapEditorView(QGraphicsView):
    """
    Custom QGraphicsView to handle zoom limits, middle-mouse panning,
    and live drawing previews correctly.
    """
    def __init__(self, scene, dialog):
        super().__init__(scene)
        self.dialog = dialog
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(80, 80, 80)))
        self.setMouseTracking(True)
        
        # Panning state
        self._panning = False
        self._pan_start = QPoint()

    def wheelEvent(self, event):
        """
        Zoom in/out, but prevent zooming out further than fitting the image to the view.
        """
        is_zooming_out = event.angleDelta().y() < 0
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        scene_rect = self.sceneRect()

        if is_zooming_out and (visible_rect.width() > scene_rect.width() or visible_rect.height() > scene_rect.height()):
            self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
            return

        factor = 1.15
        if is_zooming_out:
            factor = 1.0 / factor
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        # --- Handle Middle Mouse Panning ---
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        # --- Handle Left Mouse Drawing ---
        if event.button() == Qt.MouseButton.LeftButton:
            self.dialog.drawing = True
            scene_pos = self.mapToScene(event.pos())
            self.dialog.current_path = QPainterPath(scene_pos)
            
            # Start visual preview
            self.dialog.create_temp_path_item()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # --- Handle Panning ---
        if self._panning:
            delta = event.pos() - self._pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._pan_start = event.pos()
            event.accept()
            return

        # --- Handle Drawing ---
        scene_pos = self.mapToScene(event.pos())
        self.dialog.update_brush_cursor(scene_pos)
        
        if self.dialog.drawing and (event.buttons() & Qt.MouseButton.LeftButton):
            self.dialog.current_path.lineTo(scene_pos)
            
            # Update visual preview
            if self.dialog.temp_path_item:
                self.dialog.temp_path_item.setPath(self.dialog.current_path)
            
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # --- Stop Panning ---
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        # --- Stop Drawing ---
        if event.button() == Qt.MouseButton.LeftButton and self.dialog.drawing:
            self.dialog.drawing = False
            self.dialog.remove_temp_path_item() # Remove visual preview
            self.dialog.apply_stroke() # Burn to image
            event.accept()
        else:
            super().mouseReleaseEvent(event)

class TrimapEditorDialog(QDialog):
    def __init__(self, base_image_pil, initial_trimap_pil, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trimap Editor")
        self.setModal(True)
        self.resize(1200, 800)

        self.base_image = base_image_pil
        self.trimap_image = initial_trimap_pil.copy()
        self.final_trimap = None

        self.drawing = False
        self.current_path = QPainterPath()
        self.brush_size = 30
        self.temp_path_item = None # Visual path item

        self.init_ui()
        self.update_trimap_overlay()

    def showEvent(self, event):
        super().showEvent(event)
        if not event.spontaneous():
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        toolbar = QFrame()
        toolbar_layout = QHBoxLayout(toolbar)
        
        self.brush_group = QButtonGroup(self)
        self.rb_fg = QRadioButton("Definite Foreground"); self.rb_fg.setChecked(True)
        self.rb_bg = QRadioButton("Definite Background")
        self.rb_unknown = QRadioButton("Unknown/mixed (e.g. hair, semi transparent areas)")
        self.brush_group.addButton(self.rb_fg, 255)
        self.brush_group.addButton(self.rb_bg, 0)
        self.brush_group.addButton(self.rb_unknown, 128)
        toolbar_layout.addWidget(QLabel("<b>Brush:</b>"))
        toolbar_layout.addWidget(self.rb_fg); toolbar_layout.addWidget(self.rb_bg); toolbar_layout.addWidget(self.rb_unknown)
        toolbar_layout.addSpacing(20)

        self.lbl_brush_size = QLabel(f"Size: {self.brush_size}")
        self.slider_brush_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_brush_size.setRange(1, 400); self.slider_brush_size.setValue(self.brush_size)
        self.slider_brush_size.valueChanged.connect(self.set_brush_size)
        toolbar_layout.addWidget(self.lbl_brush_size); toolbar_layout.addWidget(self.slider_brush_size)
        toolbar_layout.addStretch()
        main_layout.addWidget(toolbar)

        self.scene = QGraphicsScene()
        self.view = TrimapEditorView(self.scene, self)
        
        bg_pixmap = pil2pixmap(self.base_image)
        self.bg_pixmap_item = QGraphicsPixmapItem(bg_pixmap)
        self.scene.addItem(self.bg_pixmap_item)
        
        self.scene.setSceneRect(self.bg_pixmap_item.boundingRect())

        self.trimap_overlay_item = QGraphicsPixmapItem()
        self.trimap_overlay_item.setOpacity(0.6)
        self.scene.addItem(self.trimap_overlay_item)
        
        self.brush_cursor_item = self.scene.addEllipse(0, 0, 0, 0, QPen(Qt.GlobalColor.white, 2, Qt.PenStyle.DashLine))
        self.brush_cursor_item.setZValue(100)
        
        main_layout.addWidget(self.view)

        button_layout = QHBoxLayout()
        btn_load = QPushButton("Load Trimap")
        btn_load.clicked.connect(self.load_trimap)
        btn_save = QPushButton("Save Trimap")
        btn_save.clicked.connect(self.save_trimap)
        btn_clear = QPushButton("Clear to Background")
        btn_clear.clicked.connect(self.clear_trimap)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton("OK"); btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept_changes)
        
        button_layout.addWidget(btn_load)
        button_layout.addWidget(btn_save)
        button_layout.addWidget(btn_clear)
        button_layout.addStretch()
        button_layout.addWidget(btn_cancel); button_layout.addWidget(btn_ok)
        main_layout.addLayout(button_layout)
        
    def create_temp_path_item(self):
        """Creates a visual path item to show where the user is currently drawing."""
        if self.temp_path_item:
            self.remove_temp_path_item()
            
        brush_id = self.brush_group.checkedId()
        # Set color based on brush type for visual feedback
        if brush_id == 255: color = QColor(255, 255, 255, 180) # White
        elif brush_id == 0: color = QColor(0, 0, 0, 180)       # Black
        else: color = QColor(0, 0, 255, 180)                   # Blue

        pen = QPen(color, self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.temp_path_item = self.scene.addPath(self.current_path, pen)
        self.temp_path_item.setZValue(50)

    def remove_temp_path_item(self):
        if self.temp_path_item and self.temp_path_item.scene():
            self.scene.removeItem(self.temp_path_item)
        self.temp_path_item = None

    def set_brush_size(self, value):
        self.brush_size = value
        self.lbl_brush_size.setText(f"Size: {self.brush_size}")
        self.update_brush_cursor(self.view.mapToScene(self.view.mapFromGlobal(self.view.cursor().pos())))

    def update_trimap_overlay(self):
        trimap_np = np.array(self.trimap_image)
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[0]   = [0, 0, 0, 255]       # Background -> Black
        lut[128] = [0, 0, 255, 255]     # Unknown -> Blue
        lut[255] = [255, 255, 255, 255] # Foreground -> White
        trimap_color_np = lut[trimap_np]
        self.trimap_overlay_item.setPixmap(numpy_to_pixmap(trimap_color_np))

    def update_brush_cursor(self, scene_pos):
        radius = self.brush_size / 2.0
        self.brush_cursor_item.setRect(scene_pos.x() - radius, scene_pos.y() - radius, self.brush_size, self.brush_size)
        
    def apply_stroke(self):
        if self.current_path.isEmpty(): return
        
        stroke_img = QImage(self.trimap_image.width, self.trimap_image.height, QImage.Format.Format_Grayscale8)
        stroke_img.fill(0)

        painter = QPainter(stroke_img)
        painter.setPen(QPen(QColor(255, 255, 255), self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.drawPath(self.current_path)
        painter.end()

        ptr = stroke_img.constBits()
        ptr.setsize(stroke_img.sizeInBytes())
        
        # Get the width of each row in memory, including padding
        bytes_per_line = stroke_img.bytesPerLine()
        
        # Create a 2D numpy array view of the buffer using the padded width
        raw_array = np.array(ptr, copy=False).reshape(stroke_img.height(), bytes_per_line)
        
        # Slice the array to remove the padding bytes from the end of each line
        # and create a new, contiguous copy of the data.
        stroke_mask_np = raw_array[:, :stroke_img.width()].copy()

        trimap_np = np.array(self.trimap_image)
        # Apply the brush value (Foreground/Background/Unknown) where the stroke was drawn
        trimap_np[stroke_mask_np > 0] = self.brush_group.checkedId()
        self.trimap_image = Image.fromarray(trimap_np)
        
        self.update_trimap_overlay()
        self.current_path = QPainterPath()

    def clear_trimap(self):
        self.trimap_image.paste(0, (0, 0, self.trimap_image.width, self.trimap_image.height))
        self.update_trimap_overlay()

    def accept_changes(self):
        self.final_trimap = self.trimap_image
        self.accept()

    def load_trimap(self):
        """Opens a file dialog to load a trimap image."""
        fname, _ = QFileDialog.getOpenFileName(self, "Load Trimap", "", "Images (*.png *.jpg *.bmp)")
        if fname:
            try:
                loaded_image = Image.open(fname)
                if loaded_image.size != self.base_image.size:
                    QMessageBox.warning(self, "Size Mismatch",
                                        "The loaded trimap's dimensions do not match the base image.")
                    return
                
                self.trimap_image = loaded_image.convert("L")
                self.update_trimap_overlay()

            except Exception as e:
                QMessageBox.critical(self, "Error Loading Trimap", f"Could not load the image:\n{e}")

    def save_trimap(self):
        """Opens a file dialog to save the current trimap image."""
        fname, _ = QFileDialog.getSaveFileName(self, "Save Trimap", "trimap.png", "PNG Image (*.png)")
        if fname:
            if not fname.lower().endswith('.png'):
                fname += '.png'
            try:
                self.trimap_image.save(fname)
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Trimap", f"Could not save the image:\n{e}")