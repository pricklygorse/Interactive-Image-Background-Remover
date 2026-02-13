from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QScrollArea, QPushButton, QHBoxLayout, QMessageBox, QApplication

from PyQt6.QtCore import Qt


import cv2
from src.utils import apply_tone_sharpness
from src.ui_dialogs import InpaintingDialog
from src.constants import PAINT_BRUSH_SCREEN_SIZE

from PIL import Image
import numpy as np
import os


class AdjustTab(QScrollArea):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWidgetResizable(True)
        
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
        
        self.init_ui()

    def init_ui(self):
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(0) 

        lbl_desc = QLabel("Colour and tone adjustments applied to the image before mask generation.")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        layout.addSpacing(10)


        

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
            slider.valueChanged.connect(lambda _: self.controller.adjust_timer.start())
            
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
        btn_rot_l.clicked.connect(lambda: self.controller.rotate_image(direction="left"))
        
        btn_rot_r = QPushButton("Rotate Right ↷")
        btn_rot_r.setToolTip("This will reset Undo history")
        btn_rot_r.clicked.connect(lambda: self.controller.rotate_image(direction="right"))
        
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
        self.btn_inpaint.clicked.connect(self.controller.open_inpainting_dialog)
        layout.addWidget(self.btn_inpaint)

        layout.addSpacing(10)

        crop_layout = QHBoxLayout()
        
        self.btn_toggle_crop = QPushButton("Crop Image ✂")
        self.btn_toggle_crop.setCheckable(True)
        self.btn_toggle_crop.toggled.connect(self.toggle_crop_mode)
        self.btn_toggle_crop.setToolTip("Draw a box on the input view to crop the original image.\n"
                                        "Note: This is a destructive action and will clear undo history.")
        
        self.btn_apply_crop = QPushButton("Apply Crop")
        self.btn_apply_crop.clicked.connect(self.controller.apply_crop)
        self.btn_apply_crop.setEnabled(False) # Disabled until mode is active
        self.btn_apply_crop.setStyleSheet("background-color: #2e7d32; color: white;") # Green to signify action
        self.btn_apply_crop.hide()

        crop_layout.addWidget(self.btn_toggle_crop)
        crop_layout.addWidget(self.btn_apply_crop)
        
        layout.addLayout(crop_layout)

        

        layout.addStretch()

        self.controller.adjust_timer.timeout.connect(self.controller.apply_adjustments)




        self.setWidget(container)
    


    def reset_adjustment_sliders(self):
        self.controller.adjust_timer.stop()
        self.controller.output_refresh_timer.stop() 

        for param, (_, _, default) in self.adj_slider_params.items():
            self.adj_sliders[param].blockSignals(True)
            self.adj_sliders[param].setValue(default)
            self.adj_sliders[param].blockSignals(False)
        
        # Update the labels and the image
        for slider in self.adj_sliders.values():
            slider.valueChanged.emit(slider.value())
        
        self.controller.apply_adjustments()

    def get_adjustment_params(self):
        return {param: slider.value() for param, slider in self.adj_sliders.items()}

    def toggle_crop_mode(self, checked):
        self.controller.crop_mode = checked
        
        if checked:
            self.controller.view_input.setCursor(Qt.CursorShape.CrossCursor)
            self.btn_toggle_crop.setText("Cancel Crop")
            self.controller.status_label.setText("CROP MODE | Drag to draw crop area | Click 'Apply Crop' to finish")
            self.btn_apply_crop.show()
            self.btn_apply_crop.setEnabled(True)
            self.controller.act_paint_mode.setEnabled(False)
            if self.controller.act_paint_mode.isChecked():
                self.controller.act_paint_mode.setChecked(False)
            
            # Disable SAM interactions visually
            self.controller.combo_sam.setEnabled(False)
            
        else:
            self.btn_toggle_crop.setText("Crop Image ✂")
            self.controller.view_input.setCursor(Qt.CursorShape.ArrowCursor)
            self.controller.status_label.setText("Ready")
            self.btn_apply_crop.hide()
            self.btn_apply_crop.setEnabled(False)
            self.controller.act_paint_mode.setEnabled(True)
            self.controller.combo_sam.setEnabled(True)
            
            # Hide the rect in the view
            if hasattr(self.controller.view_input, 'crop_rect_item'):
                self.controller.view_input.crop_rect_item.hide()