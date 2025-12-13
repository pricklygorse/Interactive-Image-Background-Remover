from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QCheckBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QDialog, QScrollArea, 
                             QGraphicsRectItem, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import (QColor, QPen, QBrush)

from PIL import Image, ImageEnhance, ImageFilter
import math
import numpy as np

from src.utils import sigmoid, pil2pixmap



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
