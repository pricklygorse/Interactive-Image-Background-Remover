from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                             QProgressBar, QTextEdit, QFileDialog, QGroupBox, QListWidget,
                             QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from src.image_session import ImageSession
from src.utils import (
    apply_tone_sharpness, sanitise_filename_for_windows,
    get_current_crop_bbox, compose_final_image, refine_mask
)

def process_batch_image(file_path, output_dir, model_manager, 
                        generation_settings, render_settings, adj_settings,
                        export_settings):
    """
    Full pipeline for processing a single image in batch mode.
    """
    try:

        session = ImageSession(file_path).load()
        
        # 2. Apply Adjustments (Tone/Sharpness)
        # Use adjustment_source_np from session for better performance
        processed_np = apply_tone_sharpness(session.source_image_np, adj_settings)
        # Convert back to PIL
        working_orig_image = Image.fromarray(cv2.cvtColor(processed_np, cv2.COLOR_BGRA2RGBA))
        
        # 3. Run Inference (Auto Model)
        model_name = generation_settings.get("model_name")
        provider_data = generation_settings.get("provider_data")
        use_2step = generation_settings.get("use_2step", False)
        
        if not model_name or "Select" in model_name:
             return False, "No model selected"

        inference_session, load_t = model_manager.get_auto_session(model_name, provider_data)
        prov_code = provider_data[2]
        
        if use_2step:
            mask_arr, status = model_manager.run_auto_inference_2step(inference_session, working_orig_image, model_name, load_t, prov_code)
        else:
            mask_arr, status = model_manager.run_auto_inference(inference_session, working_orig_image, model_name, load_t, prov_code)

        base_mask = Image.fromarray(mask_arr, mode="L")
        
        # 4. Refine Mask
        final_mask = refine_mask(base_mask, working_orig_image, generation_settings, model_manager)
        
        # 5. Compose Final Image
        final_image = compose_final_image(working_orig_image, final_mask, render_settings, model_manager)
        
        # 6. Save
        fname = os.path.basename(file_path)
        name, ext = os.path.splitext(fname)
        
        fmt = export_settings.get("format", "png")
        quality = export_settings.get("quality", 90)
        save_mask = export_settings.get("save_mask", False)
        trim = export_settings.get("trim", False)
        
        ext_map = {"png": ".png", "webp_lossless": ".webp", "webp_lossy": ".webp", "jpeg": ".jpg"}
        out_ext = ext_map.get(fmt, ".png")
        
        out_name = sanitise_filename_for_windows(name + "_nobg" + out_ext)
        out_path = os.path.join(output_dir, out_name)
        
        # Prepare for save (Handle trim)
        if trim:
            shadow_cfg = render_settings.get("shadow", {})
            drop_shadow = shadow_cfg.get("enabled", False)
            sl_s_x = shadow_cfg.get("x", 0)
            sl_s_y = shadow_cfg.get("y", 0)
            sl_s_r = shadow_cfg.get("radius", 0)
            
            bbox = get_current_crop_bbox(final_mask, drop_shadow, sl_s_x, sl_s_y, sl_s_r)
            if bbox:
                final_image = final_image.crop(bbox)
                if save_mask: final_mask = final_mask.crop(bbox) # trim mask too? usually yes

        # Handle JPEG background (White)
        if fmt == "jpeg":
             bg = Image.new("RGB", final_image.size, (255, 255, 255))
             bg.paste(final_image, mask=final_image.split()[3])
             final_image = bg

        save_params = {}
        if session.image_exif: save_params['exif'] = session.image_exif
        
        if fmt == "jpeg": save_params['quality'] = quality
        elif fmt == "webp_lossy": save_params['quality'] = quality
        elif fmt == "webp_lossless": save_params['lossless'] = True
        elif fmt == "png": save_params['optimize'] = True
        
        final_image.save(out_path, **save_params)
        
        if save_mask:
            mask_out_name = sanitise_filename_for_windows(name + "_mask.png")
            mask_out_path = os.path.join(output_dir, mask_out_name)
            final_mask.save(mask_out_path)
            
        return True, f"Saved {out_name}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e)

class BatchWorker(QThread):
    file_started = pyqtSignal(int)
    file_finished = pyqtSignal(int, bool, str)
    finished_all = pyqtSignal()
    
    def __init__(self, image_paths, output_dir, model_manager, 
                 generation_settings, render_settings, adj_settings, export_settings):
        super().__init__()
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.model_manager = model_manager
        self.generation_settings = generation_settings
        self.render_settings = render_settings
        self.adj_settings = adj_settings
        self.export_settings = export_settings
        self.is_cancelled = False

    def run(self):
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths):
            if self.is_cancelled: break
            
            self.file_started.emit(i)
            
            success, msg = process_batch_image(
                path, self.output_dir, self.model_manager,
                self.generation_settings, self.render_settings, 
                self.adj_settings, self.export_settings
            )
            
            self.file_finished.emit(i, success, msg)
        
        self.finished_all.emit()

    def cancel(self):
        self.is_cancelled = True

class BatchProcessingDialog(QDialog):
    def __init__(self, parent, image_paths, model_manager, 
                 generation_settings, render_settings, adj_settings, export_settings):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.resize(1000, 600)
        
        self.image_paths = image_paths
        self.model_manager = model_manager
        self.generation_settings = generation_settings
        self.render_settings = render_settings
        self.adj_settings = adj_settings
        self.export_settings = export_settings
        
        # Default output dir to the dir of the first image
        if self.image_paths:
            self.output_dir = os.path.dirname(self.image_paths[0])
        else:
            self.output_dir = ""
            
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # Description Label
        description_label = QLabel("Batch editor uses the values you set in the main window and applies to all the currently loaded images")
        main_layout.addWidget(description_label)

        main_layout.addSpacing(10)
        
        # Splitter Layout: Left (Files) | Right (Settings)
        content_layout = QHBoxLayout()
        
        # --- LEFT PANEL ---
        left_panel = QVBoxLayout()
        
        # 1. Image List Info
        info_grp = QGroupBox(f"Images to Process ({len(self.image_paths)})")
        ig_layout = QVBoxLayout(info_grp)
        self.list_widget = QListWidget()
        self.list_widget.addItems([os.path.basename(p) for p in self.image_paths])
        ig_layout.addWidget(self.list_widget)
        left_panel.addWidget(info_grp, stretch=1) 
        
        # 2. Output Directory Selector
        out_layout = QHBoxLayout()
        self.lbl_out = QLabel(f"Output: {self.output_dir}")
        self.lbl_out.setWordWrap(True)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_output)
        out_layout.addWidget(self.lbl_out, 1)
        out_layout.addWidget(btn_browse)
        left_panel.addLayout(out_layout)
        
        # 3. Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.image_paths))
        left_panel.addWidget(self.progress_bar)
        
        content_layout.addLayout(left_panel, stretch=2) 
        
        # --- RIGHT PANEL ---
        right_panel = QVBoxLayout()
        
        summary_grp = QGroupBox("Applied Settings")
        sum_layout = QVBoxLayout(summary_grp)
        
        summary_text = self.get_settings_summary()
        self.lbl_summary = QLabel(summary_text)
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        sum_layout.addWidget(self.lbl_summary)
        sum_layout.addStretch() 
        
        right_panel.addWidget(summary_grp)
        
        content_layout.addLayout(right_panel, stretch=1) 

        main_layout.addLayout(content_layout)
        
        # 5. Buttons
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Batch")
        self.btn_start.clicked.connect(self.start_batch)
        self.btn_cancel = QPushButton("Close")
        self.btn_cancel.clicked.connect(self.reject) 
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_cancel)
        main_layout.addLayout(btn_layout)

    def on_file_started(self, index):
        item = self.list_widget.item(index)
        item.setText(f"⏳ {os.path.basename(self.image_paths[index])} (Processing...)")
        self.list_widget.scrollToItem(item)

    def on_file_finished(self, index, success, msg):
        item = self.list_widget.item(index)
        if success:
            item.setText(f"✅ {os.path.basename(self.image_paths[index])}")
            item.setToolTip(f"Saved: {msg}")
        else:
            item.setText(f"❌ {os.path.basename(self.image_paths[index])} (Error)")
            item.setToolTip(f"Error: {msg}")
        
        self.progress_bar.setValue(index + 1)

    def start_batch(self):
        if not self.image_paths:
             QMessageBox.warning(self, "No Images", "No images loaded to process.")
             return
        
        if not os.path.isdir(self.output_dir):
             QMessageBox.warning(self, "Invalid Directory", "Please select a valid output directory.")
             return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setText("Stop")
        self.btn_cancel.clicked.disconnect()
        self.btn_cancel.clicked.connect(self.stop_batch)
        
        # Reset list items
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setText(os.path.basename(self.image_paths[i]))
            self.list_widget.item(i).setToolTip("")
        self.progress_bar.setValue(0)

        self.worker = BatchWorker(
            self.image_paths, self.output_dir, self.model_manager,
            self.generation_settings, self.render_settings, 
            self.adj_settings, self.export_settings
        )
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.finished_all.connect(self.on_finished)
        self.worker.start()

    def stop_batch(self):
        if hasattr(self, 'worker'):
            self.worker.cancel()
            self.btn_cancel.setEnabled(False) 

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setText("Close")
        self.btn_cancel.setEnabled(True)
        self.btn_cancel.clicked.disconnect()
        self.btn_cancel.clicked.connect(self.accept)
        QMessageBox.information(self, "Done", "Batch Processing Complete.")
    
    def get_settings_summary(self):
        s = []
        # Generation
        gen = self.generation_settings
        model_name = gen.get('model_name', 'None')
        s.append(f"<b>Model:</b> {model_name}")
        
        if gen.get('matting', {}).get('enabled'):
             algo = gen['matting'].get('algorithm', 'Unknown')
             limit = gen['matting'].get('longest_edge_limit', 'Unknown')
             s.append(f"<b>Matting:</b> Enabled ({algo}, {limit}px)")
        
        if gen.get('use_2step'):
            s.append("<b>Mode:</b> 2-Step (Find -> Detail)")

        # Render
        ren = self.render_settings
        
        if ren.get('clean_alpha'):
            s.append("<b>Refinement:</b> Clean Transparency (Edges)")
        
        # Adjustments
        adj = self.adj_settings
        adj_list = []
        # Defaults based on backgroundremoval.py logic
        defaults = {
            'highlight': 100, 'midtone': 100, 'shadow': 100,
            'tone_curve': 10, 'brightness': 100, 'contrast': 100,
            'saturation': 100, 'white_balance': 6500,
            'unsharp_amount': 0
        }
        
        for k, v in adj.items():
            if k in defaults and v != defaults[k]:
                adj_list.append(f"{k.replace('_', ' ').capitalize()}: {v}")
            # unsharp amount 0 check handled by default dict
        
        if adj_list:
             s.append(f"<b>Adjustments:</b><br>&nbsp;&nbsp;{', '.join(adj_list)}")

        
        bg = ren.get('background', {}).get('type', 'Transparent')
        if "Color" in bg and bg != "Transparent":
             bg += f" ({ren.get('background', {}).get('color')})"
        elif "Blur" in bg:
             bg += f" (Radius: {ren.get('background', {}).get('blur_radius')})"
             
        s.append(f"<b>Background:</b> {bg}")
        
        effects = []
        if ren.get('shadow', {}).get('enabled'): effects.append("Drop Shadow")
        if ren.get('outline', {}).get('enabled'): effects.append("Outline")
        if ren.get('inner_glow', {}).get('enabled'): effects.append("Inner Glow")
        if ren.get('tint', {}).get('enabled'): effects.append("Subject Tint")
        if ren.get('foreground_correction', {}).get('enabled'): effects.append("FG Colour Correct")
        
        if effects:
            s.append(f"<b>Effects:</b> {', '.join(effects)}")
            
        # Export
        exp = self.export_settings
        fmt = exp.get('format', 'png')
        quality_txt = ""
        if fmt in ['jpeg', 'webp_lossy']:
            quality_txt = f" (Quality: {exp.get('quality')})"
            
        s.append(f"<b>Export:</b> {fmt.upper()}{quality_txt}")
        if exp.get('trim'): s.append("<b>Trimming:</b> Enabled (Auto-Crop transparent pixels)")
        if exp.get('save_mask'): s.append("<b>Save Mask:</b> Yes")
        
        return "<br>".join(s)

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if d:
            self.output_dir = d
            self.lbl_out.setText(f"Output: {self.output_dir}")
