import os
import sys
import requests
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QProgressBar, QScrollArea, QWidget, QMessageBox, QFrame, 
                             QSizePolicy,QStackedWidget,QCheckBox)
from PyQt6.QtCore import (QThread, pyqtSignal, Qt)

REMBG_BASE_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
SAM2_BASE_URL = "https://huggingface.co/mabote-itumeleng/ONNX-SAM2-Segment-Anything/resolve/main/"

MODEL_DOWNLOAD_GROUPS = [
    
    {
        "group_name": "Segment Anything Models (SAM)",
        "type": "sam",
        "models": [
            {   "id": "mobile_sam",
                "name": "MobileSAM (Recommended)",
                "description": "Smallest, fastest SAM model, typically for mobile or low-resource devices",
                "files": [
                    {"file": "mobile_sam.encoder.onnx", "url": REMBG_BASE_URL + "mobile_sam.encoder.onnx", "size_mb": 26.9},
                    {"file": "mobile_sam.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.decoder.onnx", "size_mb": 15.7}
                ]
            },
            {   "id": "sam_vit_b_01ec64.quant",
                "name": "SAM ViT-B (Quantised)",
                "description": "Base SAM model, quantised for a good balance of accuracy and reduced size.",
                "files": [
                    {"file": "sam_vit_b_01ec64.quant.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.encoder.quant.onnx", "size_mb": 104.0},
                    {"file": "sam_vit_b_01ec64.quant.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.decoder.quant.onnx", "size_mb": 8.35}
                ]
            },
            {   "id": "sam_vit_b_01ec64",
                "name": "SAM ViT-B (Full)",
                "description": "Base SAM model, full precision for maximum accuracy with reasonable speed.",
                "files": [
                    {"file": "sam_vit_b_01ec64.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.encoder.onnx", "size_mb": 343.0},
                    {"file": "sam_vit_b_01ec64.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.decoder.onnx", "size_mb": 15.7}
                ]
            },
            {   "id": "sam_vit_l_0b3195.quant",
                "name": "SAM ViT-L (Quantised)",
                "description": "Large SAM model, quantised for improved accuracy over ViT-B with a moderate size increase. Requires both encoder and decoder.",
                "files": [
                    {"file": "sam_vit_l_0b3195.quant.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.encoder.quant.onnx", "size_mb": 317.0},
                    {"file": "sam_vit_l_0b3195.quant.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.decoder.quant.onnx", "size_mb": 8.35}
                ]
            },
            {   "id": "sam_vit_l_0b3195",
                "name": "SAM ViT-L (Full)",
                "description": "Large SAM model, full precision for highest accuracy but with the largest size. Requires both encoder and decoder.",
                "files": [
                    {"file": "sam_vit_l_0b3195.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.encoder.onnx", "size_mb": 1177.6}, 
                    {"file": "sam_vit_l_0b3195.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.decoder.onnx", "size_mb": 15.7}
                ]
            },
            {   "id": "sam2.1_hiera_tiny",
                "name": "SAM2.1 Hiera Tiny",
                "description": "SAM 2.1 Hiera model (Tiny). Fastest, but retains good accuracy.",
                "files": [
                    {"file": "sam2.1_hiera_tiny.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_tiny_encoder.onnx?download=true", "size_mb": 134.0},
                    {"file": "sam2.1_hiera_tiny.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_tiny_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
            {   "id": "sam2.1_hiera_small",
                "name": "SAM2.1 Hiera Small",
                "description": "SAM 2.1 Hiera model (Small).",
                "files": [
                    {"file": "sam2.1_hiera_small.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_small_encoder.onnx?download=true", "size_mb": 163.0},
                    {"file": "sam2.1_hiera_small.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_small_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
            {   "id": "sam2.1_hiera_base",
                "name": "SAM2.1 Hiera Base+",
                "description": "SAM 2.1 Hiera model (Base+).",
                "files": [
                    {"file": "sam2.1_hiera_base_plus.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_base_plus_encoder.onnx?download=true", "size_mb": 340.0},
                    {"file": "sam2.1_hiera_base_plus.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_base_plus_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
            {   "id": "sam2.1_hiera_large",
                "name": "SAM2.1 Hiera Large",
                "description": "SAM 2.1 Hiera model (Large).",
                "files": [
                    {"file": "sam2.1_hiera_large.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_large_encoder.onnx?download=true", "size_mb": 889.0},
                    {"file": "sam2.1_hiera_large.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_large_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
        ]
    },
    {
        "group_name": "Automatic Whole-Image Models",
        "type": "automatic_general",
        "models": [
            {   "id": "isnet-general-use",
                "name": "ISNet-General (recommended)",
                "description": "A high-performance pre-trained model for general-purpose background removal. Fast",
                "files": [
                    {"file": "isnet-general-use.onnx", "url": REMBG_BASE_URL + "isnet-general-use.onnx", "size_mb": 170.0}
                ]
            },
            {   "id": "rmbg1_4",
                "name": "BRIA RMBG 1.4 (recommended)",
                "description": "Enhancement of isnet with proprietary dataset. Fast",
                "files": [
                    {"file": "rmbg1_4.onnx", "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx?download=true", "size_mb": 176.0}
                ]
            },
            {   "id": "BiRefNet-general-epoch_244",
                "name": "BiRefNet-General",
                "description": "A pre-trained BiRefNet model for general use cases.",
                "files": [
                    {"file": "BiRefNet-general-epoch_244.onnx", "url": REMBG_BASE_URL + "BiRefNet-general-epoch_244.onnx", "size_mb": 928.0}
                ]
            },
            {   "id": "BiRefNet-general-bb_swin_v1_tiny-epoch_232",
                "name": "BiRefNet-General-Lite",
                "description": "A lightweight BiRefNet model for faster general use cases.",
                "files": [
                    {"file": "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", "url": REMBG_BASE_URL + "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx", "size_mb": 214.0}
                ]
            },
            {   "id": "u2net",
                "name": "U2Net",
                "description": "The original general-purpose background removal model. Very fast",
                "files": [
                    {"file": "u2net.onnx", "url": REMBG_BASE_URL + "u2net.onnx", "size_mb": 168.0}
                ]
            },
            {   "id": "u2net_silueta",
                "name": "Silueta (U2Net Reduced)",
                "description": "Essentially u2net but the size is significantly reduced (42.1Mb). Very fast.",
                "files": [
                    {"file": "u2net_silueta.onnx", "url": REMBG_BASE_URL + "silueta.onnx", "size_mb": 42.1}
                ]
            },
            {   "id": "u2netp",
                "name": "U2NetP (Tiny)",
                "description": "A ultralight version of the U2Net model (4.36 MB). Very fast.",
                "files": [
                    {"file": "u2netp.onnx", "url": REMBG_BASE_URL + "u2netp.onnx", "size_mb": 4.36}
                ]
            },
            {   "id": "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420",
                "name": "BiRefNet-Massive",
                "description": "BiRefNet model trained on a massive dataset (TR_DIS5K_TR_TEs) for broad coverage.",
                "files": [
                    {"file": "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx", "url": REMBG_BASE_URL + "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx", "size_mb": 928.0}
                ]
            },
            {   "id": "bria-rmbg-2.0",
                "name": "BRIA RMBG-2.0",
                "description": "Dichotomous image segmentation model for high-accuracy background removal, built on the BiRefNet architecture. Slow, heavy RAM usage",
                "files": [
                    {"file": "bria-rmbg-2.0.onnx", "url": REMBG_BASE_URL + "bria-rmbg-2.0.onnx", "size_mb": 977.0}
                ]
            },
            {   "id": "BiRefNet_HR-general-epoch_130",
                "name": "BiRefNet-HR-General",
                "description": "A BiRefNet model for general use, specifically trained on high-resolution images (e.g., 2048x2048) for excellent high-resolution performance. Slow, extremely heavy RAM usage",
                "files": [
                    {"file": "BiRefNet_HR-general-epoch_130.onnx", "url": REMBG_BASE_URL + "BiRefNet_HR-general-epoch_130.onnx", "size_mb": 1044.48}
                ]
            },
            {   "id": "BiRefNet_HR-matting-epoch_135",
                "name": "BiRefNet-HR-Matting",
                "description": "A BiRefNet model for general matting use, trained on high-resolution images (e.g., 2048x2048) for high-quality matting and edge detail.",
                "files": [
                    {"file": "BiRefNet_HR-matting-epoch_135.onnx", "url": REMBG_BASE_URL + "BiRefNet_HR-matting-epoch_135.onnx", "size_mb": 1044.48}
                ]
            },
            {   "id": "BiRefNet-portrait-epoch_150",
                "name": "BiRefNet-Portrait",
                "description": "BiRefNet model specifically tuned for segmenting human portraits.",
                "files": [
                    {"file": "BiRefNet-portrait-epoch_150.onnx", "url": REMBG_BASE_URL + "BiRefNet-portrait-epoch_150.onnx", "size_mb": 928.0}
                ]
            },
            {   "id": "u2net_human_seg",
                "name": "U2Net-Human-Seg",
                "description": "A pre-trained U2Net model specifically for human segmentation.",
                "files": [
                    {"file": "u2net_human_seg.onnx", "url": REMBG_BASE_URL + "u2net_human_seg.onnx", "size_mb": 168.0}
                ]
            },
            {   "id": "isnet-anime",
                "name": "ISNet-Anime",
                "description": "A high-accuracy segmentation model specifically for anime characters.",
                "files": [
                    {"file": "isnet-anime.onnx", "url": REMBG_BASE_URL + "isnet-anime.onnx", "size_mb": 168.0}
                ]
            },
            {   "id": "BiRefNet-COD-epoch_125",
                "name": "BiRefNet-COD",
                "description": "BiRefNet model for Concealed Object Detection (COD).",
                "files": [
                    {"file": "BiRefNet-COD-epoch_125.onnx", "url": REMBG_BASE_URL + "BiRefNet-COD-epoch_125.onnx", "size_mb": 928.0}
                ]
            },
            {   "id": "BiRefNet-HRSOD_DHU-epoch_115",
                "name": "BiRefNet-HRSOD",
                "description": "BiRefNet model for High-Resolution Salient Object Detection (HRSOD).",
                "files": [
                    {"file": "BiRefNet-HRSOD_DHU-epoch_115.onnx", "url": REMBG_BASE_URL + "BiRefNet-HRSOD_DHU-epoch_115.onnx", "size_mb": 928.0}
                ]
            },
            {   "id": "BiRefNet-DIS-epoch_590",
                "name": "BiRefNet-DIS",
                "description": "BiRefNet model for Dichotomous Image Segmentation (DIS).",
                "files": [
                    {"file": "BiRefNet-DIS-epoch_590.onnx", "url": REMBG_BASE_URL + "BiRefNet-DIS-epoch_590.onnx", "size_mb": 928.0}
                ]
            },
            {   "id": "BEN2",
                "name": "BEN2: Background Erase Network (2025)",
                "description": "BEN2 (Background Erase Network) introduces a novel approach to foreground segmentation through its innovative Confidence Guided Matting (CGM) pipeline",
                "files": [
                    {"file": "ben2_base.onnx", "url": "https://huggingface.co/PramaLLC/BEN2/resolve/main/BEN2_Base.onnx?download=true", "size_mb": 223.0}  # mirror https://huggingface.co/RedbeardNZ/BEN2/resolve/main/BEN2_Base.onnx?download=true
                ]
            },
            {   "id": "mvanet",
                "name": "MVANet (2024)",
                "description": "Multi-view Aggregation Network for Dichotomous Image Segmentation",
                "files": [
                    {"file": "mvanet_full.onnx", "url": "https://huggingface.co/onnx-community/MVANet-ONNX/resolve/main/onnx/model.onnx?download=true", "size_mb": 422.0} 
                ]
            },
            {   "id": "mvanet_quant",
                "name": "MVANet (2024) quantised",
                "description": "Multi-view Aggregation Network for Dichotomous Image Segmentation",
                "files": [
                    {"file": "mvanet_quant.onnx", "url": "https://huggingface.co/onnx-community/MVANet-ONNX/resolve/main/onnx/model_quantized.onnx?download=true", "size_mb": 141.0} 
                ]
            },
        ]
    },
    {
        "group_name": "Edge Refinement Models",
        "type": "edge",
        "models": [
            {   "id": "vitmatte_small_composition",
                "name": "ViTMatte Small (Composition 1k dataset)",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_s_composition.onnx", "url":"https://huggingface.co/Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model.onnx?download=true", "size_mb": 104.0},
                ]
            },
            {   "id": "vitmatte_small_composition_quant",
                "name": "ViTMatte Small (Composition 1k dataset) Quantised",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_s_composition_quant.onnx", "url":"https://huggingface.co/Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model_quantized.onnx?download=true", "size_mb": 27.5},
                ]
            },
        ]
    },
]


# --- QThread for Background Downloading ---
class ModelDownloadThread(QThread):
    # Signals for file-level updates: (filename: str, progress: int)
    progress_updated = pyqtSignal(str, int) 
    # Signals for download finished: (filename: str, status: str)
    download_finished = pyqtSignal(str, str) 
    # Signals for error: (filename: str, error_message: str)
    error_occurred = pyqtSignal(str, str) 

    def __init__(self, file_data, model_root_dir, parent=None):
        super().__init__(parent)
        self.file_data = file_data
        self.model_root_dir = model_root_dir
        self._is_running = True

    def run(self):
        filename = self.file_data['file']
        url = self.file_data['url']
        save_path = os.path.join(self.model_root_dir, filename)
        temp_path = save_path + ".download" 

        try:
            # Use stream=True to handle large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                self.error_occurred.emit(filename, "Download failed: Server did not provide file size.")
                return

            block_size = 8192
            bytes_downloaded = 0

            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            with open(temp_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if not self._is_running:
                        f.close()
                        if os.path.exists(save_path): os.remove(save_path) 
                        self.download_finished.emit(filename, "Cancelled")
                        return

                    f.write(data)
                    bytes_downloaded += len(data)
                    progress = int(bytes_downloaded * 100 / total_size)
                    self.progress_updated.emit(filename, progress)
            
            os.replace(temp_path, save_path)
            self.download_finished.emit(filename, "Success")

        except requests.exceptions.RequestException as e:
            self.error_occurred.emit(filename, f"Network Error: {e}")
        except Exception as e:
            if os.path.exists(temp_path): 
                try: os.remove(temp_path)
                except: pass
            self.error_occurred.emit(filename, f"File System Error: {e}")

    def stop(self):
        self._is_running = False

# --- QDialog for the Download Manager UI ---
class ModelDownloadDialog(QDialog):
    def __init__(self, model_root_dir, main_app_instance=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Download Manager")
        
        self.setMinimumSize(800, 600) 
        self.resize(800, 600) 
        
        self.layout = QVBoxLayout(self)
        
        self.model_root_dir = model_root_dir
        self.main_app_instance = main_app_instance
        self.settings = main_app_instance.settings if main_app_instance else None
        
        self.download_threads = {} 
        self.row_widgets = {} 
        self.file_progress_tracker = {}

        self.status_label = QLabel("Ready to download models. Setting large models to <i>Load on Startup</i> is not advised")
        self.layout.addWidget(self.status_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.list_widget = QWidget()
        self.list_layout = QVBoxLayout(self.list_widget)
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.list_widget)
        self.layout.addWidget(scroll)
        
        for group in MODEL_DOWNLOAD_GROUPS:
            self._add_group_header(group['group_name'])
            model_type = group.get('type')
            for model in group['models']:
                self._add_model_row(model, model_type)

    def _add_group_header(self, name):
        header_lbl = QLabel(f"<b>--- {name} ---</b>")
        header_lbl.setStyleSheet("padding-top: 10px; padding-bottom: 5px;")
        header_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.list_layout.addWidget(header_lbl)

    def _check_if_model_downloaded(self, model_data):
        for f in model_data['files']:
            path = os.path.join(self.model_root_dir, f['file'])
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
        return True

    def _add_model_row(self, model_data, model_type):
        model_name = model_data['name']
        model_id = model_data['id']
        is_downloaded = self._check_if_model_downloaded(model_data)
        combined_size = sum(f['size_mb'] for f in model_data['files'])

        row = QFrame()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(5, 5, 5, 5)

        # Top row: Name, Size, Button, Status
        h_top = QHBoxLayout()
        name_lbl = QLabel(f"<b>{model_name}</b> ({combined_size:.1f} MB)")
        name_lbl.setToolTip(model_data['description'])
        
        name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) 
        h_top.addWidget(name_lbl)

        # Use a QStackedWidget to swap between a download button and a progress bar.
        action_widget = QStackedWidget()
        action_widget.setFixedSize(120, 30)

        download_btn = QPushButton("Download")
        download_btn.clicked.connect(lambda _, m=model_data: self._start_model_group_download(m))

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setTextVisible(True)

        action_widget.addWidget(download_btn)
        action_widget.addWidget(progress_bar)

        if is_downloaded:
            progress_bar.setValue(100)
            progress_bar.setFormat("Downloaded")
            action_widget.setCurrentWidget(progress_bar)
        else:
            action_widget.setCurrentWidget(download_btn)

        h_top.addWidget(action_widget)
        
        row_layout.addLayout(h_top)

        # Second row: Description and Startup Checkbox
        h_bottom = QHBoxLayout()
        desc_lbl = QLabel(model_data['description'])
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet("font-size: 9pt; color: #666;")
        h_bottom.addWidget(desc_lbl)

        startup_cb = QCheckBox("Load on Startup")
        startup_cb.setVisible(is_downloaded)
        startup_cb.toggled.connect(
            lambda checked, mid=model_id, mtype=model_type: self._on_startup_box_toggled(checked, mid, mtype)
        )

        # Set initial checked state from settings
        if self.settings and model_type and is_downloaded:
            setting_key = f"startup_{model_type}_model"
            if self.settings.value(setting_key) == model_id:
                startup_cb.setChecked(True)

        h_bottom.addWidget(startup_cb, 0, Qt.AlignmentFlag.AlignRight)
        row_layout.addLayout(h_bottom)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        row_layout.addWidget(divider)
        
        self.list_layout.addWidget(row)
        self.row_widgets[model_id] = {
            'row': row, 
            'progress': progress_bar, 
            'button': download_btn,
            'action_widget': action_widget,
            'startup_cb': startup_cb,
            'model_type': model_type,
            'model_name': model_name
        }

    def _on_startup_box_toggled(self, is_checked, model_id, model_type):
        if not self.settings or not model_type: return
        setting_key = f"startup_{model_type}_model"

        if is_checked:
            # Uncheck any other checkbox of the same type
            for other_id, widgets in self.row_widgets.items():
                if widgets['model_type'] == model_type and other_id != model_id:
                    widgets['startup_cb'].setChecked(False)
            self.settings.setValue(setting_key, model_id)
        else:
            # If the user unchecks the currently selected one, clear the setting
            if self.settings.value(setting_key) == model_id:
                self.settings.remove(setting_key)

    def _start_model_group_download(self, model_data):
        model_name = model_data['name']
        
        # Cleanup previous threads
        if model_name in self.download_threads:
            for thread in self.download_threads[model_name]:
                if thread.isRunning(): thread.stop(); thread.wait()
            del self.download_threads[model_name]
        
        self.download_threads[model_name] = []
        self.file_progress_tracker[model_name] = {}
        
        # Update UI
        w = self.row_widgets[model_data['id']]
        w['action_widget'].setCurrentWidget(w['progress'])
        w['progress'].setValue(0)
        w['progress'].setFormat("%p%")
        self.status_label.setText(f"Starting download for {model_name}...")

        # Start threads for all files
        for i, file_data in enumerate(model_data['files']):
            thread = ModelDownloadThread(file_data, self.model_root_dir)
            
            # Signal: (filename_sig: str, progress_sig: int)
            thread.progress_updated.connect(
                lambda filename_sig, progress_sig, m=model_name: self._update_model_progress(m, filename_sig, progress_sig)
            )
            # Signal: (filename_sig: str, msg_sig: str)
            thread.error_occurred.connect(
                lambda filename_sig, msg_sig, m=model_name: self._handle_model_error(m, filename_sig, msg_sig)
            )
            # Signal: (filename_sig: str, status_sig: str)
            thread.download_finished.connect(
                lambda filename_sig, status_sig, m=model_name: self._handle_file_finish(m, filename_sig, status_sig)
            )
            
            self.download_threads[model_name].append(thread)
            self.file_progress_tracker[model_name][file_data['file']] = 0
            thread.start()

    def _update_model_progress(self, model_name, filename, progress):
        
        self.file_progress_tracker[model_name][filename] = progress
        
        total_files = len(self.file_progress_tracker[model_name])
        
        sum_progress = sum(self.file_progress_tracker[model_name].values())
        
        overall_progress = int(sum_progress / total_files)
        
        w = next((widgets for widgets in self.row_widgets.values() if widgets['model_name'] == model_name), None)
        if w:
            w['progress'].setValue(overall_progress)


    def _handle_model_error(self, model_name, filename, message):
        w = next((widgets for widgets in self.row_widgets.values() if widgets['model_name'] == model_name), None)
        if w:
            w['action_widget'].setCurrentWidget(w['button'])
            w['button'].setText("Retry")
            w['button'].setToolTip(f"Failed: {message}. Click to try again.")
            w['button'].setEnabled(True)
            w['progress'].setValue(0)
            QMessageBox.critical(self, "Download Error", f"Failed to download {model_name} ({filename}): {message}")
            self.status_label.setText(f"Download FAILED for {model_name}.")

    def _handle_file_finish(self, model_name, filename, status):
        
        if status == "Cancelled": return 

        # The thread has finished successfully, update its final progress.
        self.file_progress_tracker[model_name][filename] = 100

        # Update the overall progress
        self._update_model_progress(model_name, filename, 100) 
        
        # Check if ALL files for this model are finished
        all_finished = all(p == 100 for p in self.file_progress_tracker[model_name].values())
        
        w = next((widgets for widgets in self.row_widgets.values() if widgets['model_name'] == model_name), None)
        if w and all_finished:
            w['progress'].setValue(100)
            w['progress'].setFormat("Downloaded")
            self.status_label.setText(f"Successfully downloaded all files for {model_name}.")
            w['startup_cb'].setVisible(True)

            # Refresh model lists in the main application
            if self.main_app_instance and hasattr(self.main_app_instance, 'populate_sam_models'):
                self.main_app_instance.populate_sam_models()
                self.main_app_instance.populate_whole_models()
                self.main_app_instance.populate_matting_models()
                if hasattr(self.main_app_instance, 'update_cached_model_icons'):
                    self.main_app_instance.update_cached_model_icons()
        elif w:
             # Progress will be updated by _update_model_progress as other files download
             pass

    def closeEvent(self, event):
        # Stop all running threads if the dialog is closed
        for threads in self.download_threads.values():
            for thread in threads:
                if thread.isRunning():
                    thread.stop()
                    thread.wait()
        event.accept()