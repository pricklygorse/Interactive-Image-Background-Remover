import os
import sys
import requests
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QProgressBar, QScrollArea, QWidget, QMessageBox, QFrame, 
                             QSizePolicy,QStackedWidget,QCheckBox, QComboBox,QLineEdit, QFileDialog, QSlider)
from PyQt6.QtWidgets import QTabWidget
from PyQt6.QtCore import (QThread, pyqtSignal, Qt, QSize)

REMBG_BASE_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
SAM2_BASE_URL = "https://huggingface.co/mabote-itumeleng/ONNX-SAM2-Segment-Anything/resolve/main/"
DEEPFILL_BASE_URL = 'https://storage.googleapis.com/ailia-models/deepfillv2/'

MODEL_DOWNLOAD_GROUPS = [
    
    {
        "group_name": "Interactive (SAM) Models",
        "type": "sam",
        "models": [
            {   "id": "mobile_sam",
                "name": "MobileSAM (Recommended)",
                "description": "Smallest, fastest Segment Anything model, typically for mobile or low-resource devices",
                "files": [
                    {"file": "mobile_sam.encoder.onnx", "url": REMBG_BASE_URL + "mobile_sam.encoder.onnx", "size_mb": 26.9},
                    {"file": "mobile_sam.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.decoder.onnx", "size_mb": 15.7}
                ]
            },
            {   "id": "sam_vit_b_01ec64.quant",
                "name": "SAM ViT-B (Quantised)",
                "description": "Base Segment Anything model, quantised for a good balance of accuracy and reduced size.",
                "files": [
                    {"file": "sam_vit_b_01ec64.quant.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.encoder.quant.onnx", "size_mb": 104.0},
                    {"file": "sam_vit_b_01ec64.quant.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.decoder.quant.onnx", "size_mb": 8.35}
                ]
            },
            {   "id": "sam_vit_b_01ec64",
                "name": "SAM ViT-B (Full)",
                "description": "Base Segment Anything model, full precision for maximum accuracy with reasonable speed.",
                "files": [
                    {"file": "sam_vit_b_01ec64.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.encoder.onnx", "size_mb": 343.0},
                    {"file": "sam_vit_b_01ec64.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_b_01ec64.decoder.onnx", "size_mb": 15.7}
                ]
            },
            {   "id": "sam_vit_l_0b3195.quant",
                "name": "SAM ViT-L (Quantised)",
                "description": "Large Segment Anything model, quantised for improved accuracy over ViT-B with a moderate size increase. Requires both encoder and decoder.",
                "files": [
                    {"file": "sam_vit_l_0b3195.quant.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.encoder.quant.onnx", "size_mb": 317.0},
                    {"file": "sam_vit_l_0b3195.quant.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.decoder.quant.onnx", "size_mb": 8.35}
                ]
            },
            {   "id": "sam_vit_l_0b3195",
                "name": "SAM ViT-L (Full)",
                "description": "Large Segment Anything model, full precision for highest accuracy but with the largest size. Requires both encoder and decoder.",
                "files": [
                    {"file": "sam_vit_l_0b3195.encoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.encoder.onnx", "size_mb": 1177.6}, 
                    {"file": "sam_vit_l_0b3195.decoder.onnx", "url": REMBG_BASE_URL + "sam_vit_l_0b3195.decoder.onnx", "size_mb": 15.7}
                ]
            },
            {   "id": "sam2.1_hiera_tiny",
                "name": "SAM2.1 Hiera Tiny",
                "description": "Segment Anything 2.1 Hiera model (Tiny). Fastest, but retains good accuracy.",
                "files": [
                    {"file": "sam2.1_hiera_tiny.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_tiny_encoder.onnx?download=true", "size_mb": 134.0},
                    {"file": "sam2.1_hiera_tiny.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_tiny_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
            {   "id": "sam2.1_hiera_small",
                "name": "SAM2.1 Hiera Small",
                "description": "Segment Anything 2.1 Hiera model (Small).",
                "files": [
                    {"file": "sam2.1_hiera_small.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_small_encoder.onnx?download=true", "size_mb": 163.0},
                    {"file": "sam2.1_hiera_small.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_small_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
            {   "id": "sam2.1_hiera_base",
                "name": "SAM2.1 Hiera Base+",
                "description": "Segment Anything 2.1 Hiera model (Base+).",
                "files": [
                    {"file": "sam2.1_hiera_base_plus.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_base_plus_encoder.onnx?download=true", "size_mb": 340.0},
                    {"file": "sam2.1_hiera_base_plus.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_base_plus_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
            {   "id": "sam2.1_hiera_large",
                "name": "SAM2.1 Hiera Large",
                "description": "Segment Anything 2.1 Hiera model (Large).",
                "files": [
                    {"file": "sam2.1_hiera_large.encoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_large_encoder.onnx?download=true", "size_mb": 889.0},
                    {"file": "sam2.1_hiera_large.decoder.onnx", "url": SAM2_BASE_URL + "sam2.1_hiera_large_decoder.onnx?download=true", "size_mb": 20.7}
                ]
            },
        ]
    },
    {
        "group_name": "Automatic Models",
        "type": "automatic_general",
        "models": [
            {   "id": "isnet-general-use",
                "name": "ISNet-General (recommended) (2022)",
                "description": "A high-performance pre-trained model for general-purpose background removal. Fast",
                "files": [
                    {"file": "isnet-general-use.onnx", "url": REMBG_BASE_URL + "isnet-general-use.onnx", "size_mb": 170.0}
                ]
            },
            {   "id": "rmbg1_4",
                "name": "BRIA RMBG 1.4 (recommended) (2023)",
                "description": "Enhancement of isnet with proprietary dataset. Fast",
                "files": [
                    {"file": "rmbg1_4.onnx", "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx?download=true", "size_mb": 176.0}
                ]
            },
            {   "id": "BiRefNet-general-epoch_244",
                "name": "BiRefNet-General (2024)",
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
                "name": "U2Net (2020)",
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
                "description": "A ultralight, very fast, version of the U2Net model (4.36 MB)",
                "files": [
                    {"file": "u2netp.onnx", "url": REMBG_BASE_URL + "u2netp.onnx", "size_mb": 4.36}
                ]
            },
            {   "id": "modnet",
                "name": "Modnet Portrait Matting (2022)",
                "description": "Very fast, lightweight model for portraits.",
                "files": [
                    {"file": "modnet_portrait.onnx", "url": "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx?download=true", "size_mb": 25.9}
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
                "name": "BRIA RMBG-2.0 (2024)",
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
                "name": "ViTMatte Small - Composition 1k dataset",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_s_composition.onnx", "url":"https://huggingface.co/Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model.onnx?download=true", "size_mb": 104.0},
                ]
            },
            {   "id": "vitmatte_small_composition_quant",
                "name": "ViTMatte Small - Composition 1k dataset Quantised (recommended)",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_s_composition_quant.onnx", "url":"https://huggingface.co/Xenova/vitmatte-small-composition-1k/resolve/main/onnx/model_quantized.onnx?download=true", "size_mb": 27.5},
                ]
            },
            {   "id": "vitmatte_base_composition",
                "name": "ViTMatte Base - Composition 1k dataset",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_b_composition.onnx", "url":"https://huggingface.co/Xenova/vitmatte-base-composition-1k/resolve/main/onnx/model.onnx?download=true", "size_mb": 387.0},
                ]
            },
            # {   "id": "vitmatte_base_composition_quant",   # base quants produce broken masks, or something i'm doing wrong
            #     "name": "ViTMatte Base - Composition 1k dataset Quantised",
            #     "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
            #     "files": [
            #         {"file": "vitmatte_b_composition_quant.onnx", "url":"https://huggingface.co/Xenova/vitmatte-base-composition-1k/resolve/main/onnx/model_quantized.onnx?download=true", "size_mb": 99.0},
            #     ]
            # },
            {   "id": "vitmatte_small_distinctions",
                "name": "ViTMatte Small - Distinctions 646 dataset",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_s_distinctions.onnx", "url":"https://huggingface.co/Xenova/vitmatte-small-distinctions-646/resolve/main/onnx/model.onnx?download=true", "size_mb": 104.0},
                ]
            },
            {   "id": "vitmatte_small_distinctions_quant",
                "name": "ViTMatte Small - Distinctions 646 dataset Quantised (recommended)",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_s_distinctions_quant.onnx", "url":"https://huggingface.co/Xenova/vitmatte-small-distinctions-646/resolve/main/onnx/model_quantized.onnx?download=true", "size_mb": 27.5},
                ]
            },
            {   "id": "vitmatte_base_distinctions",
                "name": "ViTMatte Base - Distinctions 646 dataset",
                "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
                "files": [
                    {"file": "vitmatte_b_distinctions.onnx", "url":"https://huggingface.co/Xenova/vitmatte-base-distinctions-646/resolve/main/onnx/model.onnx?download=true", "size_mb": 387.0},
                ]
            },
            # {   "id": "vitmatte_base_distinctions_quant",       # base quants produce broken masks, or something i'm doing wrong
            #     "name": "ViTMatte Base - Distinctions 646 dataset Quantised",
            #     "description": "Image Matting with Pretrained Plain Vision Transformers (2023)",
            #     "files": [
            #         {"file": "vitmatte_b_distinctions_quant.onnx", "url":"https://huggingface.co/Xenova/vitmatte-base-distinctions-646/resolve/main/onnx/model_quantized.onnx?download=true", "size_mb": 99.0},
            #     ]
            # },
            {   "id": "indexnet",
                "name": "IndexNet (2019)",
                "description": "An older image matting model. Included for completeness",
                "files": [
                    {"file": "indexnet.onnx", "url":"https://storage.googleapis.com/ailia-models/indexnet/indexnet.onnx", "size_mb": 23.8},
                ]
            },
        ]
    },
    {
    "group_name": "Inpainting Models",
    "type": "inpainting",
    "models": [
        {   "id": "lama",
            "name": "LaMa: Large Mask Inpainting (2022)",
            "description": "Resolution-robust Large Mask Inpainting with Fourier Convolutions. Excellent for removing objects.",
            "files": [
                {"file": "lama.onnx", "url": "https://storage.googleapis.com/ailia-models/lama/lama.onnx", "size_mb": 208.0}
            ]
        },
        {   "id": "migan",
            "name": "MI-GAN (fast)",
            "description": "A Simple Baseline for Image Inpainting on Mobile Devices (2023).",
            "files": [
                {"file": "migan_pipeline_v2.onnx", "url": "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx?download=true", "size_mb": 28.1}
            ]
        },
        # DeepFillv2 CelebA Models
        {   "id": "deepfillv2_celeba_256x256",
            "name": "DeepFillv2 CelebA (256x256)",
            "description": "DeepFillv2 trained on the CelebA (Faces) dataset at 256x256 resolution.",
            "files": [
                {"file": "deepfillv2_celeba_256x256.onnx", "url": DEEPFILL_BASE_URL + "deepfillv2_celeba_256x256.onnx", "size_mb": 16.9},
            ]
        },
        # DeepFillv2 Places Models ---
        {   "id": "deepfillv2_places_256x256",
            "name": "DeepFillv2 Places (256x256)",
            "description": "DeepFillv2 trained on the Places (General Scenes) dataset at 256x256 resolution.",
            "files": [
                {"file": "deepfillv2_places_256x256.onnx", "url": DEEPFILL_BASE_URL + "deepfillv2_places_256x256.onnx", "size_mb": 16.9},
            ]
        },
        {   "id": "deepfillv2_places_512x512",
            "name": "DeepFillv2 Places (512x512)",
            "description": "DeepFillv2 trained on the Places (General Scenes) dataset at 512x512 resolution.",
            "files": [
                {"file": "deepfillv2_places_512x512.onnx", "url": DEEPFILL_BASE_URL + "deepfillv2_places_512x512.onnx", "size_mb": 16.9},
            ]
        },
        {   "id": "deepfillv2_places_1024x1024",
            "name": "DeepFillv2 Places (1024x1024)",
            "description": "DeepFillv2 trained on the Places (General Scenes) dataset at 1024x1024 resolution.",
            "files": [
                {"file": "deepfillv2_places_1024x1024.onnx", "url": DEEPFILL_BASE_URL + "deepfillv2_places_1024x1024.onnx", "size_mb": 16.9},
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
class SettingsDialog(QDialog):
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

        self.status_label = QLabel("")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.layout.addWidget(self.status_label)
        

        # add tabs

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True) # Ensures consistent background rendering across OS styles
        self.layout.addWidget(self.tabs)

        self._add_general_tab()
        
        for group in MODEL_DOWNLOAD_GROUPS:
            self._add_tab(group)


    def _add_general_tab(self):
        # Use a scroll area to match the structure of the model tabs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        tab_content = QWidget()
        tab_content.setAutoFillBackground(True) # Force use of the Window colour role
        layout = QVBoxLayout(tab_content)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Theme Selection
        theme_group = QFrame()
        theme_group.setFrameShape(QFrame.Shape.StyledPanel)
        theme_layout = QHBoxLayout(theme_group)
        
        theme_label = QLabel("<b>Interface Theme:</b>")
        theme_label.setToolTip("Switch between Light and Dark mode.")
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark"])
        
        # Set current value from settings
        current_theme = self.settings.value("theme", "light") if self.settings else "light"
        self.theme_combo.setCurrentText(current_theme.capitalize())
        
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        
        theme_layout.addWidget(theme_label)
        theme_layout.addWidget(self.theme_combo)
        theme_layout.addStretch()
        
        layout.addWidget(theme_group)

        # Model Download Location
        path_group = QFrame()
        path_group.setFrameShape(QFrame.Shape.StyledPanel)
        path_layout = QVBoxLayout(path_group)
        
        path_label = QLabel("<b>Model Download Location:</b>")
        path_label.setToolTip("Select the folder where ONNX models are stored and downloaded to.")
        path_layout.addWidget(path_label)
        
        path_selection_layout = QHBoxLayout()
        self.path_display = QLineEdit(self.model_root_dir)
        self.path_display.setReadOnly(True)
        
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_model_path)
        
        path_selection_layout.addWidget(self.path_display)
        path_selection_layout.addWidget(btn_browse)
        path_layout.addLayout(path_selection_layout)
        
        layout.addWidget(path_group)


        # Paintbrush Settings
        paint_group = QFrame()
        paint_group.setFrameShape(QFrame.Shape.StyledPanel)
        paint_layout = QVBoxLayout(paint_group)
        
        paint_label = QLabel("<b>Paintbrush Behaviour:</b>")
        paint_layout.addWidget(paint_label)
        
        self.chk_paint_direct = QCheckBox("Paintbrush edits output mask directly")
        self.chk_paint_direct.setToolTip(
            "Checked: Paint strokes immediately modify the final output.\n"
            "Unchecked: Paint strokes modify the blue model overlay (requires 'Add/Sub' to commit)."
        )
        
        # Default to True
        is_direct = self.settings.value("paint_edits_working_mask", True, type=bool) if self.settings else True
        self.chk_paint_direct.setChecked(is_direct)
        self.chk_paint_direct.toggled.connect(
            lambda checked: self.settings.setValue("paint_edits_working_mask", checked) if self.settings else None
        )



        # Refinement & Matting Settings
        refine_group = QFrame()
        refine_group.setFrameShape(QFrame.Shape.StyledPanel)
        refine_layout = QVBoxLayout(refine_group)
        
        refine_label = QLabel("<b>Refinement & Matting Settings:</b>")
        refine_layout.addWidget(refine_label)

        # Matting Resolution
        res_layout = QHBoxLayout()
        res_label = QLabel("Alpha Matting Processing Resolution:")
        self.res_combo = QComboBox()
        self.res_combo.addItems(["512", "1024", "1536", "2048"])
        
        current_res = self.settings.value("matting_longest_edge", "1024") if self.settings else "1024"
        self.res_combo.setCurrentText(current_res)
        self.res_combo.currentTextChanged.connect(
            lambda v: self.settings.setValue("matting_longest_edge", v) if self.settings else None
        )
        
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.res_combo)
        res_layout.addStretch()
        refine_layout.addLayout(res_layout)
        res_desc = QLabel("Higher resolutions can result in a higher quality alpha mask, but at the cost of substantially increased processing time. 1024px recommended as this matches the output size of most mask generation models, so the quality should not be degraded")
        res_desc.setWordWrap(True)
        refine_layout.addWidget(res_desc)

        # Smart Refine Context Padding
        pad_layout = QHBoxLayout()
        pad_label = QLabel("Smart Refine PaintBrush Context Padding (Experimental):")
        pad_label.setToolTip("Determines how much 'Definite' background and foreground the AI sees around your brush stroke.\nCan have small impact on output quality")
        
        self.pad_slider = QSlider(Qt.Orientation.Horizontal)
        self.pad_slider.setRange(1, 5)
        self.pad_slider.setFixedWidth(150)
        
        current_pad = self.settings.value("smart_refine_padding", 3, type=int) if self.settings else 3
        self.pad_slider.setValue(current_pad)
        
        pad_val_label = QLabel(str(current_pad))
        pad_val_label.setFixedWidth(30)
        
        self.pad_slider.valueChanged.connect(lambda v: pad_val_label.setText(str(v)))
        self.pad_slider.valueChanged.connect(
             lambda v: self.settings.setValue("smart_refine_padding", v) if self.settings else None
        )

        pad_layout.addWidget(pad_label)
        pad_layout.addWidget(self.pad_slider)
        pad_layout.addWidget(pad_val_label)
        pad_layout.addStretch()
        refine_layout.addLayout(pad_layout)

        layout.addWidget(refine_group)

        # Guided Filter Settings
        gf_group = QFrame()
        gf_group.setFrameShape(QFrame.Shape.StyledPanel)
        gf_layout = QVBoxLayout(gf_group)
        
        gf_label = QLabel("<b>Guided Filter Settings (Smart Refine):</b>")
        gf_layout.addWidget(gf_label)

        # GF Radius
        gfr_layout = QHBoxLayout()
        gfr_label = QLabel("Radius:")
        gfr_label.setToolTip("Neighborhood size for the guided filter. Higher values sample a larger area but can cause haloing.")
        
        self.gfr_slider = QSlider(Qt.Orientation.Horizontal)
        self.gfr_slider.setRange(1, 50)
        self.gfr_slider.setFixedWidth(150)
        
        current_gfr = self.settings.value("gf_radius", 10, type=int) if self.settings else 10
        self.gfr_slider.setValue(current_gfr)
        
        gfr_val_label = QLabel(str(current_gfr))
        gfr_val_label.setFixedWidth(30)
        
        self.gfr_slider.valueChanged.connect(lambda v: gfr_val_label.setText(str(v)))
        self.gfr_slider.valueChanged.connect(
             lambda v: self.settings.setValue("gf_radius", v) if self.settings else None
        )

        gfr_layout.addWidget(gfr_label)
        gfr_layout.addWidget(self.gfr_slider)
        gfr_layout.addWidget(gfr_val_label)
        gfr_layout.addStretch()
        gf_layout.addLayout(gfr_layout)

        # GF Epsilon
        gfe_layout = QHBoxLayout()
        gfe_label = QLabel("Epsilon (Regularization):")
        gfe_label.setToolTip("Regularization parameter. Lower values (e.g. 1e-8) make the filter sharper and more edge-following. Higher values (e.g. 1e-4) are smoother.")
        
        self.gfe_slider = QSlider(Qt.Orientation.Horizontal)
        self.gfe_slider.setRange(4, 8) # representing 1e-4 to 1e-8
        self.gfe_slider.setFixedWidth(150)
        
        current_gfe_exp = self.settings.value("gf_epsilon_exponent", 6, type=int) if self.settings else 6
        self.gfe_slider.setValue(current_gfe_exp)
        
        gfe_val_label = QLabel(f"1e-{current_gfe_exp}")
        gfe_val_label.setFixedWidth(50)
        
        self.gfe_slider.valueChanged.connect(lambda v: gfe_val_label.setText(f"1e-{v}"))
        self.gfe_slider.valueChanged.connect(
             lambda v: self.settings.setValue("gf_epsilon_exponent", v) if self.settings else None
        )

        gfe_layout.addWidget(gfe_label)
        gfe_layout.addWidget(self.gfe_slider)
        gfe_layout.addWidget(gfe_val_label)
        gfe_layout.addStretch()
        gf_layout.addLayout(gfe_layout)


        # GF De-halo (Halo Suppression)
        # Cutoff
        gfhc_layout = QHBoxLayout()
        gfhc_label = QLabel("Halo Suppression Cutoff:")
        gfhc_label.setToolTip("Any alpha value below this will be set to 0. Useful for removing faint ghosting/halos.")
        
        self.gfhc_slider = QSlider(Qt.Orientation.Horizontal)
        self.gfhc_slider.setRange(0, 150)
        self.gfhc_slider.setFixedWidth(150)
        
        current_gfhc = self.settings.value("gf_halo_cutoff", 40, type=int) if self.settings else 40
        self.gfhc_slider.setValue(current_gfhc)
        
        gfhc_val_label = QLabel(str(current_gfhc))
        gfhc_val_label.setFixedWidth(30)
        
        self.gfhc_slider.valueChanged.connect(lambda v: gfhc_val_label.setText(str(v)))
        self.gfhc_slider.valueChanged.connect(
             lambda v: self.settings.setValue("gf_halo_cutoff", v) if self.settings else None
        )

        gfhc_layout.addWidget(gfhc_label)
        gfhc_layout.addWidget(self.gfhc_slider)
        gfhc_layout.addWidget(gfhc_val_label)
        gfhc_layout.addStretch()
        gf_layout.addLayout(gfhc_layout)

        # Limit
        gfhl_layout = QHBoxLayout()
        gfhl_label = QLabel("Halo Suppression Limit:")
        gfhl_label.setToolTip("Alpha values between the Cutoff and this Limit will be ramped linearly. Values above this remain untouched.")
        
        self.gfhl_slider = QSlider(Qt.Orientation.Horizontal)
        self.gfhl_slider.setRange(0, 255)
        self.gfhl_slider.setFixedWidth(150)
        
        current_gfhl = self.settings.value("gf_halo_limit", 255, type=int) if self.settings else 255
        self.gfhl_slider.setValue(current_gfhl)
        
        gfhl_val_label = QLabel(str(current_gfhl))
        gfhl_val_label.setFixedWidth(30)
        
        self.gfhl_slider.valueChanged.connect(lambda v: gfhl_val_label.setText(str(v)))
        self.gfhl_slider.valueChanged.connect(
             lambda v: self.settings.setValue("gf_halo_limit", v) if self.settings else None
        )

        gfhl_layout.addWidget(gfhl_label)
        gfhl_layout.addWidget(self.gfhl_slider)
        gfhl_layout.addWidget(gfhl_val_label)
        gfhl_layout.addStretch()
        gf_layout.addLayout(gfhl_layout)

        layout.addWidget(gf_group)






        
        paint_layout.addWidget(self.chk_paint_direct)
        layout.addWidget(paint_group)
        
        scroll.setWidget(tab_content)
        self.tabs.addTab(scroll, "Settings")

    def _on_theme_changed(self, theme_text):
        if self.main_app_instance:
            self.main_app_instance.set_theme(theme_text.lower())

    def _on_browse_model_path(self):
        new_dir = QFileDialog.getExistingDirectory(
            self, "Select Model Directory", self.model_root_dir
        )
        if new_dir:
            # Normalise path for cross-platform consistency
            new_dir = os.path.normpath(new_dir)
            self.model_root_dir = new_dir
            self.path_display.setText(new_dir)
            
            if self.settings:
                self.settings.setValue("model_root_dir", new_dir)
            
            if self.main_app_instance:
                self.main_app_instance.update_model_root_dir(new_dir)
            
            self._refresh_model_status()
                
            QMessageBox.information(self, "Model Directory Changed", 
                                    "The model directory has been updated. You may need to move your existing models to the new folder or re-download them." "Restart recommended")

    def _add_tab(self, group):
        # Scroll area uses 'Base' colour; tab_content must auto-fill with 'Window' colour to match
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        tab_content = QWidget()
        tab_content.setAutoFillBackground(True) 
        tab_layout = QVBoxLayout(tab_content)
        tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(tab_content)
        self.tabs.addTab(scroll, group['group_name'])

        model_type = group.get('type')
        for model in group['models']:
            self._add_model_row(model, model_type, tab_layout)

    def _check_if_model_downloaded(self, model_data):
        for f in model_data['files']:
            path = os.path.join(self.model_root_dir, f['file'])
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
        return True

    def _add_model_row(self, model_data, model_type, list_layout):
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
        name_lbl.setMinimumWidth(200)
        h_top.addWidget(name_lbl)

        # Use a QStackedWidget to swap between a download button and a progress bar.
        action_widget = QStackedWidget()
        action_widget.setFixedSize(120, 30)

        download_btn = QPushButton("Download")
        download_btn.clicked.connect(lambda _, m=model_data: self._start_model_group_download(m))

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        progress_bar.setMinimumSize(QSize(100, 0))

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
        desc_lbl.setMinimumWidth(200)
        desc_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        f = desc_lbl.font()
        f.setPointSize(11)
        desc_lbl.setFont(f)
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

        list_layout.addWidget(row)
        self.row_widgets[model_id] = {
            'row': row,
            'progress': progress_bar,
            'button': download_btn,
            'action_widget': action_widget,
            'startup_cb': startup_cb,
            'model_type': model_type,
            'model_name': model_name,
            'model_data': model_data
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
    
    def _refresh_model_status(self):
        """Updates the Download/Downloaded status for all rows based on current path."""
        for model_id, widgets in self.row_widgets.items():
            model_data = widgets.get('model_data')
            if not model_data: continue
            
            is_downloaded = self._check_if_model_downloaded(model_data)
            if is_downloaded:
                widgets['progress'].setValue(100)
                widgets['progress'].setFormat("Downloaded")
                widgets['action_widget'].setCurrentWidget(widgets['progress'])
                widgets['startup_cb'].setVisible(True)
            else:
                widgets['action_widget'].setCurrentWidget(widgets['button'])
                widgets['startup_cb'].setVisible(False)
                # Ensure startup setting is cleared if model is no longer available
                widgets['startup_cb'].setChecked(False)


    def closeEvent(self, event):
        # Stop all running threads if the dialog is closed
        for threads in self.download_threads.values():
            for thread in threads:
                if thread.isRunning():
                    thread.stop()
                    thread.wait()
        event.accept()