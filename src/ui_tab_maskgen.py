from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QCheckBox, QScrollArea, QSpacerItem, 
                             QSizePolicy, QRadioButton, QButtonGroup, QFrame)
from PyQt6.QtCore import Qt
from src.ui_widgets import CollapsibleFrame
from src.model_manager import ModelManager



class MaskGenTab(QScrollArea):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWidgetResizable(True)
        self.init_ui()

    def init_ui(self):

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
        self.btn_download.clicked.connect(self.controller.open_settings)
        layout.addWidget(self.btn_download)

        lbl_models_download = QLabel("Models are run on the current view. Zoom and position your subject for best results")
        lbl_models_download.setWordWrap(True)
        layout.addWidget(lbl_models_download)
        
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
        self.combo_sam.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_sam.setMinimumContentsLength(1)
        self.combo_sam.currentTextChanged.connect(lambda t: self.controller.settings.setValue("last_sam_model", t) if t else None)
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
        self.combo_whole.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_whole.setMinimumContentsLength(1)
        self.combo_whole.currentTextChanged.connect(lambda t: self.controller.settings.setValue("last_auto_model", t) if t else None)
        layout.addWidget(self.combo_whole)
        
        # Run Model Button and layout adjustment
        h_whole_model = QHBoxLayout()
        self.btn_whole = QPushButton("Run Model (M)"); self.btn_whole.clicked.connect(lambda: self.controller.run_automatic_model())
        self.btn_whole.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                border: 1px solid #005a9e;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #c8c8c8;
                color: #a1a1a1;
                border: 1px solid #c8c8c8;
            }
        """)
        h_whole_model.addWidget(self.combo_whole)
        h_whole_model.addWidget(self.btn_whole)
        layout.addLayout(h_whole_model)

        self.chk_2step_auto = QCheckBox("2-Step (Find Subject -> Re-run)")
        self.chk_2step_auto.setToolTip("Runs the model twice: first on the entire image to find the subject,\n "
                                       "then a second time on a zoomed crop of the subject for maximum detail."
                                       "\n\nNOTE: This is different to typical usage, where the model is run on the view that you are currently zoomed in to,\n"
                                       "which may instead be the optimal way to get a higher quality mask")
        layout.addWidget(self.chk_2step_auto)

        layout.addSpacing(20)

        lbl_mat_gen = QLabel("<b>DRAW AND REFINE (Matting)</b>")
        #lbl_mat_gen.setToolTip("Draw a mask and let the model calculate the difficult areas.")
        layout.addWidget(lbl_mat_gen)
        
        lbl_mat_note = QLabel("Draw a rough initial mask/trimap and let models calculate the tricky bits such as hair.")
        lbl_mat_note.setWordWrap(True)
        lbl_mat_note.setToolTip("If you have a mask generated from another model, the editor will inherit this as a starting point.")
        layout.addWidget(lbl_mat_note)

        h_mat_btn_layout = QHBoxLayout()
        
        self.btn_open_trimap_gen = QPushButton("1. Draw Trimap")
        self.btn_open_trimap_gen.clicked.connect(self.controller.open_trimap_editor)
        self.btn_open_trimap_gen.setToolTip("Open the editor to draw where the foreground and background are.")

        h_mat_algo_layout = QHBoxLayout()
        h_mat_algo_layout.addWidget(QLabel("Matting Model:"))
        
        self.combo_matting_gen = QComboBox()
        self.combo_matting_gen.setToolTip("Select the specialized matting model to use.")
        self.combo_matting_gen.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo_matting_gen.setMinimumContentsLength(1)
        h_mat_algo_layout.addWidget(self.combo_matting_gen)
        layout.addLayout(h_mat_algo_layout)
        
        self.btn_run_matting_gen = QPushButton("2. Run Matting")
        self.btn_run_matting_gen.setToolTip("Runs the selected matting model using your hand-drawn trimap.")
        self.btn_run_matting_gen.clicked.connect(self.controller.run_matting_on_custom_trimap)
        self.btn_run_matting_gen.setStyleSheet("font-weight: bold; background-color: #444; color: white;")
        
        h_mat_btn_layout.addWidget(self.btn_open_trimap_gen)
        h_mat_btn_layout.addWidget(self.btn_run_matting_gen)
        layout.addLayout(h_mat_btn_layout)

        self.chk_tiled_matting = QCheckBox("Use Tiled Matting (1:1 Detail, Slower)")
        self.chk_tiled_matting.setToolTip("Processes the image in small overlapping tiles at full resolution for a higher quality mask.\n"
                                          "Much lower VRAM usage compared to setting a larger matting resolution limit in the settings")
        self.chk_tiled_matting.setChecked(self.controller.settings.value("tiled_matting_enabled", False, type=bool))
        self.chk_tiled_matting.toggled.connect(lambda c: self.controller.settings.setValue("tiled_matting_enabled", c))
        layout.addWidget(self.chk_tiled_matting)

        layout.addSpacing(40)


        btn_clr = QPushButton("Clear SAM Clicks/ Model Masks (C)"); btn_clr.clicked.connect(self.controller.clear_overlay)
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

        last_sam = self.controller.settings.value("sam_exec_short_code", "cpu")
        idx = 0 # Default to first item
        for i in range(self.combo_sam_model_EP.count()):
            if self.combo_sam_model_EP.itemData(i)[2] == last_sam:
                idx = i
                break
        self.combo_sam_model_EP.setCurrentIndex(idx)
        self.combo_sam_model_EP.currentIndexChanged.connect(self.controller.on_sam_EP_changed)
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
        self.sam_cache_group.buttonToggled.connect(self.controller.on_sam_cache_changed)

        last_sam_cache_mode = self.controller.settings.value("sam_ram_cache_mode", 1, type=int)
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
        
        last_exec = self.controller.settings.value("exec_short_code", "cpu")
        idx = 0 # Default to first item
        for i in range(self.combo_auto_model_EP.count()):
            if self.combo_auto_model_EP.itemData(i)[2] == last_exec:
                idx = i
                break
        self.combo_auto_model_EP.setCurrentIndex(idx)
        self.combo_auto_model_EP.currentIndexChanged.connect(self.controller.on_auto_EP_changed)
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
        self.auto_cache_group.buttonToggled.connect(self.controller.on_auto_cache_changed)

        # End Hardware Acceleration
        layout.addWidget(self.hw_options_frame)
        
        # Persistence for Collapsible Frame
        hw_collapsed = self.controller.settings.value("hw_options_collapsed", True, type=bool)
        self.hw_options_frame.set_collapsed(hw_collapsed)
        self.hw_options_frame.toggled.connect(
            lambda collapsed: self.controller.settings.setValue("hw_options_collapsed", collapsed)
        )

        last_auto_cache_mode = self.controller.settings.value("auto_ram_cache_mode", 1, type=int)
        self.auto_cache_group.blockSignals(True)
        self.auto_cache_group.button(last_auto_cache_mode).setChecked(True)
        self.auto_cache_group.blockSignals(False)


        

        # End Hardware Acceleration


        layout.addStretch()
        
        self.setWidget(container)