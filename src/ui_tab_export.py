from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QCheckBox, QScrollArea, QSlider,
                             QSizePolicy, QFrame, QColorDialog)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backgroundremoval import BackgroundRemoverGUI

class ExportTab(QScrollArea):
    def __init__(self,  controller: 'BackgroundRemoverGUI'):
        super().__init__()
        self.controller = controller
        self.setWidgetResizable(True)

        self.shadow_timer = QTimer()
        self.shadow_timer.setSingleShot(True)
        self.shadow_timer.setInterval(50)  # Wait 50ms after last movement
        self.shadow_timer.timeout.connect(self.controller.update_output_preview)

        self.outline_color = QColor(0, 0, 0)
        self.inner_glow_color = QColor(255, 255, 255)
        self.tint_color = QColor(255, 200, 150)

        
        self.init_ui()

    def init_ui(self):
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
        self.combo_bg_color.currentTextChanged.connect(self.controller.handle_bg_change)
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
        self.chk_export_mask.setChecked(self.controller.settings.value("export_mask", False, type=bool))
        self.chk_export_mask.toggled.connect(lambda checked: self.controller.settings.setValue("export_mask", checked))
        layout.addWidget(self.chk_export_mask)

        self.chk_export_trim = QCheckBox("Trim Transparent Pixels (Auto-Crop)")
        self.chk_export_trim.toggled.connect(self.controller.update_output_preview)
        self.chk_export_trim.setToolTip("If exporting the global mask with the image, the mask is <b>not<b> trimmed. This is to allow you to return to editing the original image.")
        layout.addWidget(self.chk_export_trim)

        # Initialize visibility
        self.toggle_export_quality_visibility()



        layout.addSpacing(40)
        btn_qsave = QPushButton("Quick Save (JPG, White BG)"); btn_qsave.clicked.connect(lambda: self.controller.save_image(quick_save=True))
        layout.addWidget(btn_qsave)
        btn_save = QPushButton("Export Final Image"); btn_save.clicked.connect(self.controller.save_image)
        layout.addWidget(btn_save)
        btn_save_clp = QPushButton("Save to Clipboard"); btn_save_clp.clicked.connect(lambda: self.controller.save_image(clipboard=True))
        layout.addWidget(btn_save_clp)

        

        layout.addStretch()

        self.setWidget(container)



    def toggle_export_quality_visibility(self):
        """Enables or disables the quality slider based on the selected file format."""
        fmt = self.combo_export_fmt.currentData()
        is_lossy = fmt in ["webp_lossy", "jpeg"]
        self.export_quality_frame.setEnabled(is_lossy)

    def toggle_shadow_options(self, checked):
        if checked: self.shadow_frame.show()
        else: self.shadow_frame.hide()
        self.controller.update_output_preview()

    def toggle_outline_options(self, checked):
        self.outline_frame.setVisible(checked)
        self.controller.update_output_preview()

    def pick_outline_color(self):
        color = QColorDialog.getColor(self.outline_color, self, "Select Outline Colour")
        if color.isValid():
            self.outline_color = color
            self.update_outline_color_button()
            self.controller.update_output_preview()

    def update_outline_color_button(self):
        self.btn_outline_color.setText(self.outline_color.name())
        # Set text colour based on luminance for readability
        text_color = "white" if self.outline_color.lightnessF() < 0.5 else "black"
        self.btn_outline_color.setStyleSheet(
            f"background-color: {self.outline_color.name()}; color: {text_color};"
        )

    def toggle_inner_glow_options(self, checked):
        self.inner_glow_frame.setVisible(checked)
        self.controller.update_output_preview()

    def pick_inner_glow_color(self):
        color = QColorDialog.getColor(self.inner_glow_color, self, "Inner Glow Color")
        if color.isValid():
            self.inner_glow_color = color
            self.update_inner_glow_color_button()
            self.controller.update_output_preview()

    def update_inner_glow_color_button(self):
        text_color = "white" if self.inner_glow_color.lightnessF() < 0.5 else "black"
        self.btn_ig_color.setStyleSheet(f"background-color: {self.inner_glow_color.name()}; color: {text_color};")
        self.btn_ig_color.setText(self.inner_glow_color.name())

    def toggle_tint_options(self, checked):
        self.tint_frame.setVisible(checked)
        self.controller.update_output_preview()

    def pick_tint_color(self):
        color = QColorDialog.getColor(self.tint_color, self, "Tint Color")
        if color.isValid():
            self.tint_color = color
            self.update_tint_color_button()
            self.controller.update_output_preview()

    def update_tint_color_button(self):
        text_color = "white" if self.tint_color.lightnessF() < 0.5 else "black"
        self.btn_tint_color.setStyleSheet(f"background-color: {self.tint_color.name()}; color: {text_color};")
        self.btn_tint_color.setText(self.tint_color.name())

    def get_export_settings(self):
        return {
            "format": self.combo_export_fmt.currentData(),
            "quality": self.sl_export_quality.value(),
            "save_mask": self.chk_export_mask.isChecked(),
            "trim": self.chk_export_trim.isChecked()
        }