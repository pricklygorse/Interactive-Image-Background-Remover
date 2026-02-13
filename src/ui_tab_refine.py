import math
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QComboBox, QCheckBox, QScrollArea, QSlider,
                             QRadioButton, QButtonGroup, QFrame)

if TYPE_CHECKING:
    from backgroundremoval import BackgroundRemoverGUI

class RefineTab(QScrollArea):
    def __init__(self,  controller: 'BackgroundRemoverGUI'):
        super().__init__()
        self.controller = controller
        self.setWidgetResizable(True)

        self.trimap_timer = QTimer()
        self.trimap_timer.setSingleShot(True)
        self.trimap_timer.setInterval(100) 
        self.trimap_timer.timeout.connect(self.controller.update_trimap_preview)

        self.refinement_timer = QTimer()
        self.refinement_timer.setSingleShot(True)
        self.refinement_timer.setInterval(500) # 500ms delay to prevent spamming ONNX models
        self.refinement_timer.timeout.connect(lambda: self.controller.modify_mask())


        self.init_ui()

    def init_ui(self):

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
        self.chk_clean_alpha.setChecked(self.controller.settings.value("clean_alpha", True, type=bool))

        self.chk_clean_alpha.toggled.connect(
            lambda checked: self.controller.settings.setValue("clean_alpha", checked)
        )
        self.chk_clean_alpha.toggled.connect(self.controller.update_output_preview)
        layout.addWidget(self.chk_clean_alpha)

        self.chk_estimate_foreground = QCheckBox("Background Colour Bleed\n Correction (Recommended, slow)")
        self.chk_estimate_foreground.setToolTip("Recalculates the colours of the foreground cutout to remove background colours that bleed through semi transparent areas.\n"
                                                "Recommended for soft edges such as hair when overlaying onto a different coloured background.")
        layout.addWidget(self.chk_estimate_foreground)
        self.chk_estimate_foreground.setChecked(self.controller.settings.value("estimate_foreground", False, type=bool))    
        self.chk_estimate_foreground.toggled.connect(self.controller.update_output_preview)
        self.chk_estimate_foreground.toggled.connect(
            lambda checked: self.controller.settings.setValue("estimate_foreground", checked)
        )
        

        self.fg_container = QWidget()
        fg_main_layout = QVBoxLayout(self.fg_container)
        fg_main_layout.setSpacing(0)
        fg_main_layout.setContentsMargins(20, 0, 0, 0)

        # first row: label + combo
        top_row = QHBoxLayout()
        #fg_label = QLabel("FG Colour Correction:")
        self.fg_combo = QComboBox()
        self.fg_combo.setToolTip("Algorithm used to remove background colour bleed from the foreground image.")
        self.fg_combo.addItem("Multi-Level", "ml")
        self.fg_combo.addItem("Blur Fusion", "blur_fusion_2")
        self.fg_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)

        current_fg = self.controller.settings.value("fg_correction_algo", "ml") if self.controller.settings else "ml"
        idx = self.fg_combo.findData(current_fg)
        self.fg_combo.setCurrentIndex(idx if idx >= 0 else 0)
        def on_fg_algo_changed(index: int):
            data = self.fg_combo.itemData(index)
            self.radius_slider.setVisible(data != "ml")
            if self.controller.settings:
                self.controller.settings.setValue("fg_correction_algo", data)
            self.controller.update_output_preview()

        self.fg_combo.currentIndexChanged.connect(on_fg_algo_changed)
        
        #top_row.addWidget(fg_label)
        top_row.addWidget(self.fg_combo)
        #top_row.addStretch()

        R_MIN = 60
        R_MAX = 1500

        self.radius_slider = QSlider(Qt.Orientation.Horizontal)
        self.radius_slider.setRange(0, 100)
        self.radius_slider.setToolTip("Larger values sample a larger area of possible background colours to remove from the foreground.")

        # load saved radius (natural value)
        natural_radius = int(self.controller.settings.value("fg_correction_radius", 90)) if self.controller.settings else 90

        def natural_to_slider(r):
            r = max(R_MIN, min(R_MAX, r))
            t = (math.log10(r) - math.log10(R_MIN)) / (math.log10(R_MAX) - math.log10(R_MIN))
            return int(round(t * 100))

        def slider_to_natural(pos: int) -> float:
            t = pos / 100.0
            log_r = math.log10(R_MIN) + t * (math.log10(R_MAX) - math.log10(R_MIN))
            return 10 ** log_r

        # initalise slider from saved radius
        self.radius_slider.setValue(natural_to_slider(natural_radius))

        self.radius_timer = QTimer(self)
        self.radius_timer.setSingleShot(True)
        self.radius_timer.setInterval(100)

        def on_radius_slider_changed(pos: int):
            r = slider_to_natural(pos)
            if self.controller.settings:
                self.controller.settings.setValue("fg_correction_radius", int(round(r)))
            self.radius_timer.start()

        self.radius_slider.valueChanged.connect(on_radius_slider_changed)
        self.radius_timer.timeout.connect(self.controller.update_output_preview)

        top_row.addWidget(self.radius_slider)
        fg_main_layout.addLayout(top_row)
        #fg_main_layout.addWidget(self.radius_slider)

        self.fg_container.setVisible(self.chk_estimate_foreground.isChecked())
        self.radius_slider.setVisible(self.fg_combo.currentData() != "ml")
        layout.addWidget(self.fg_container)

        self.chk_estimate_foreground.toggled.connect(
            lambda checked: self.fg_container.setVisible(checked)
        )
        



        layout.addSpacing(10)

        lbl_modifiers = QLabel("<b>MODEL OUTPUT MASK REFINEMENTS</b>")
        lbl_modifiers.setContentsMargins(3, 0, 0, 0)
        layout.addWidget(lbl_modifiers)

        lbl_desc = QLabel("Applied to the AI model's output mask.")
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        #layout.addSpacing(10)

        # indent smart refine to highlight (crudely) that they are part of the live preview
        indent_container = QWidget()
        indent_layout = QVBoxLayout(indent_container)
        indent_layout.setContentsMargins(10, 0, 0, 0)

        h_expand_layout = QHBoxLayout()
        self.lbl_mask_expand = QLabel("Expand/Contract: 0")
        self.lbl_mask_expand.setMinimumWidth(120)
        self.sl_mask_expand = QSlider(Qt.Orientation.Horizontal)
        self.sl_mask_expand.setRange(-50, 50)
        self.sl_mask_expand.setValue(0)
        self.sl_mask_expand.setToolTip("Positive values expand the mask boundary, negative values shrink it.")
        self.sl_mask_expand.valueChanged.connect(lambda v: self.lbl_mask_expand.setText(f"Expand/Contract: {v}"))
        self.sl_mask_expand.valueChanged.connect(self.controller.trigger_refinement_update)
        h_expand_layout.addWidget(self.lbl_mask_expand)
        h_expand_layout.addWidget(self.sl_mask_expand)
        indent_layout.addLayout(h_expand_layout)


        self.chk_binarise_mask = QCheckBox("Remove Mask Partial Transparency")
        self.chk_binarise_mask.toggled.connect(self.controller.trigger_refinement_update)
        indent_layout.addWidget(self.chk_binarise_mask)


        self.chk_soften = QCheckBox("Soften Mask")
        soften_checked = self.controller.settings.value("soften_mask", False, type=bool)
        self.chk_soften.setChecked(soften_checked)
        self.chk_soften.toggled.connect(lambda checked: self.controller.settings.setValue("soften_mask", checked))
        self.chk_soften.toggled.connect(self.controller.trigger_refinement_update)
        indent_layout.addWidget(self.chk_soften)
        
        indent_layout.addSpacing(20)


        matt_tt = "Runs a second pass with a matting model to estimate the transparency of mask edges.\n" + "This can improve the quality of detailed edges such as hair, especially when using binary mask models like SAM.\n" + "This requires a trimap (a foreground, unknown, background mask), either estimated from a SAM or automatic models, or manually drawn.\n" + "Alpha matting is computationally expensive"
        lbl_alpha = QLabel("<b>SMART REFINE</b>")
        lbl_alpha.setToolTip(matt_tt)
        indent_layout.addWidget(lbl_alpha)
        lbl_alpha_desc = QLabel("Runs the model output through a specialist model to refine difficult areas like hair and fur. ViTMatte is recommended")
        lbl_alpha_desc.setWordWrap(True)
        lbl_alpha_desc.setToolTip(matt_tt)
        indent_layout.addWidget(lbl_alpha_desc)

        btn_refine_download = QPushButton("Download Refinement Models 📥")
        btn_refine_download.setToolTip("Download Refinement Models. VitMatte Small is recommended")
        btn_refine_download.clicked.connect(self.controller.open_settings)
        indent_layout.addWidget(btn_refine_download)


        ma_layout = QHBoxLayout()

        matting_label = QLabel("Algorithm:")
        matting_tt = "Additional models can be downloaded using the model manager.\nThe default included PyMatting algo can be very slow on large images.\nViTMatte (model downloader) can be much faster and far more accurate"
        matting_label.setToolTip(matting_tt)
        ma_layout.addWidget(matting_label, 0)


        self.combo_matting_algorithm = QComboBox()
        self.combo_matting_algorithm.setToolTip(matting_tt)
        self.combo_matting_algorithm.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_matting_algorithm.setMinimumContentsLength(1)
        self.combo_matting_algorithm.currentIndexChanged.connect(self.controller.trigger_refinement_update)
        ma_layout.addWidget(self.combo_matting_algorithm, 1)

        indent_layout.addLayout(ma_layout)

        self.chk_alpha_matting = QCheckBox("Enable Smart Refine")
        self.chk_alpha_matting.setToolTip(matt_tt)
        self.chk_alpha_matting.toggled.connect(self.controller.handle_alpha_matting_toggle)
        self.chk_alpha_matting.toggled.connect(self.controller.trigger_refinement_update)
        indent_layout.addWidget(self.chk_alpha_matting)


        # --- Alpha Matting Options Frame ---
        self.alpha_matting_frame = QFrame()
        am_layout = QVBoxLayout(self.alpha_matting_frame)
        am_layout.setContentsMargins(15, 5, 0, 5) # Indent options slightly

        self.chk_show_trimap = QCheckBox("Show Trimap on Input")
        self.chk_show_trimap.setToolTip("Displays the generated trimap on the input view.\n"
                                        "White = Foreground, Blue = Unknown (semi transparent, e.g. hair edges), Black = Background")
        self.chk_show_trimap.toggled.connect(self.controller.toggle_trimap_display)
        am_layout.addWidget(self.chk_show_trimap)

        # --- Trimap Source Radio Buttons ---
        lbl_tri_src = QLabel("<b>Trimap Source:</b>")
        lbl_tri_src.setToolTip("A trimap is a guidance mask that specifies what is definite foreground, definite background, and 'unknown/mixed' for the model to calculate")
        am_layout.addWidget(lbl_tri_src)
        self.trimap_mode_group = QButtonGroup(self)
        self.rb_trimap_auto = QRadioButton("Automatic\n(expand fg/bg edge)")
        self.rb_trimap_auto.setToolTip("Expands the border between the foreground and background to create region for the model to refine the mask.")
        self.rb_trimap_custom = QRadioButton("Custom (user-drawn)")
        
        self.trimap_mode_group.addButton(self.rb_trimap_auto)
        self.trimap_mode_group.addButton(self.rb_trimap_custom)
        
        self.rb_trimap_auto.setChecked(True)

        am_layout.addWidget(self.rb_trimap_auto)
        
        self.trimap_mode_group.buttonToggled.connect(self.controller.on_trimap_mode_changed)
        self.trimap_mode_group.buttonToggled.connect(self.controller.trigger_refinement_update)
        
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
                                                                            self.trimap_timer.start(),
                                                                            self.controller.trigger_refinement_update()))
            h_layout.addWidget(label)
            h_layout.addWidget(slider)
            return label, slider, h_layout

        self.lbl_fg_erode, self.sl_fg_erode, fg_layout = make_am_slider_row("FG Shrink", 0, 200, 30)
        self.sl_fg_erode.setToolTip("Shrinks the solid foreground area to create the 'unknown' region.")
        sliders_layout.addLayout(fg_layout)

        self.lbl_bg_erode, self.sl_bg_erode, bg_layout = make_am_slider_row("BG Shrink", 0, 200, 30)
        self.sl_bg_erode.setToolTip("Shrinks the solid background area to create the 'unknown' region.")
        sliders_layout.addLayout(bg_layout)

        am_layout.addWidget(self.auto_trimap_sliders_widget)

        am_layout.addWidget(self.rb_trimap_custom)
        
        self.btn_edit_trimap = QPushButton("Open Trimap Editor...")
        self.btn_edit_trimap.clicked.connect(self.controller.open_trimap_editor)
        self.btn_edit_trimap.setEnabled(False)
        am_layout.addWidget(self.btn_edit_trimap)
        
        


        indent_layout.addWidget(self.alpha_matting_frame)
        self.alpha_matting_frame.hide()

        # end alpha matting


        layout.addWidget(indent_container)

        layout.addStretch()

        self.setWidget(container)

    