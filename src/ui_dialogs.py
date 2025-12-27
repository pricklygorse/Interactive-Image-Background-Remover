from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QDialog, QScrollArea, 
                             QGraphicsRectItem, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import (QColor, QPen, QBrush,QImage, QPixmap)

from PIL import Image
import math
import numpy as np

from timeit import default_timer as timer
import cv2


from src.utils import numpy_bgra_to_pixmap, apply_tone_sharpness


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
