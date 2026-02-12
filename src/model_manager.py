import os
import gc
import cv2
import sys
import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from timeit import default_timer as timer
from pymatting import estimate_alpha_sm, estimate_foreground_ml, estimate_foreground_cf
from src.utils import estimate_fg_blur_fusion

from .constants import SAM_TRT_WARMUP_POINTS


class ModelManager:
    def __init__(self, model_root_dir, cache_root_dir):
        self.model_root_dir = model_root_dir
        self.cache_root_dir = cache_root_dir
        
        # Cache Containers
        self.loaded_automatic_models = {}
        self.loaded_sam_models = {}
        self.loaded_matting_models = {}
        
        # SAM State
        self.sam_encoder = None
        self.sam_decoder = None
        self.sam_model_path = None
        self.encoder_output = None
        self.last_crop_rect = None # For optimization
        self.last_enc_shape = None
        self.last_transform = None
        self.last_orig_size = None
        
        # Settings state (defaults)
        self.auto_cache_mode = 1 # 0=None, 1=Last, 2=All
        self.sam_cache_mode = 1
        self._sam_trt_warmed = set()


    @staticmethod
    def get_available_ep_options():
        """
        Returns list of (Display Name, ProviderStr, OptionsDict, ShortCode)
        """
        try:
            available = ort.get_available_providers()
            print(available)
        except:
            print("No onnxruntime providers")
            return []
        options = []
        
        # Generic Vulkan compatible provider. Requires a custom build of onnxruntime with --use_webgpu
        if "WebGpuExecutionProvider" in available:
            options.append(("WebGPU (Experimental)", "WebGpuExecutionProvider", {}, "webgpu"))
        
        # DEBUG override to show all providers
        #available = ["CUDAExecutionProvider", "CPUExecutionProvider", "TensorrtExecutionProvider", "OpenVINOExecutionProvider", "CoreMLExecutionProvider"] # DEBUG
        
        if "TensorrtExecutionProvider" in available:
            # We will generate specific cache paths at runtime, 
            # so we pass an empty dict here, or default options.
            options.append(("TensorRT (GPU)", "TensorrtExecutionProvider", {}, "trt"))

        if "CUDAExecutionProvider" in available:
            options.append(("CUDA (GPU)", "CUDAExecutionProvider", {}, "cuda"))

        # Windows generic provider
        if "DmlExecutionProvider" in available:
            options.append(("DirectML (GPU)", "DmlExecutionProvider", {}, "dml"))

        if "OpenVINOExecutionProvider" in available:
            try:
                ov_devices = ort.capi._pybind_state.get_available_openvino_device_ids()
            except Exception:
                ov_devices = []
            
            # Fallback if query fails but provider exists
            if not ov_devices: 
                ov_devices = ['CPU']

            for dev in ov_devices:
                options.append((f"OpenVINO-{dev}", "OpenVINOExecutionProvider", {'device_type': dev}, f"ov-{dev.lower()}"))
        if "CoreMLExecutionProvider" in available:
            options.append(("CoreML", "CoreMLExecutionProvider", {}, "coreml"))

        # CPU always available
        options.append(("CPU", "CPUExecutionProvider", {}, "cpu"))

        return options

    def check_is_cached(self, model_name, provider_short_code):
        if not os.path.isdir(self.cache_root_dir): return False
        if provider_short_code == 'cpu': return False
                
        # Simple check: Look for folder containing model_name and provider code
        # This allows "trt" to match "TensorrtExecutionProvider_u2net"
        
        sanitised_model_name = "".join([c for c in model_name if c.isalnum() or c in "-_"])
        
        for folder in os.listdir(self.cache_root_dir):
            full_path = os.path.join(self.cache_root_dir, folder)
            if not os.path.isdir(full_path): continue
            
            if sanitised_model_name in folder and len(os.listdir(full_path)) > 0:
                # Distinguish between providers
                if provider_short_code == 'trt' and 'Tensorrt' in folder: return True
                if provider_short_code.startswith('ov') and 'OpenVINO' in folder:
                    # Check specific device match (e.g. ov-gpu)
                    device = provider_short_code.split('-')[1].upper()
                    if device in folder: return True
                
        return False
    
    def _create_inference_session(self, model_path, provider_str, provider_options, model_id_name):
        """
        Generic builder for ONNX sessions. 
        Handles cache directory creation automatically based on provider + model name.
        """

        # Make path absolute so ONNX external data is resolved from the model's folder,
        # not from the process working directory.
        model_path = os.path.abspath(model_path)

        # fix sam2 on openvino. Doesn't like dynamic batch dimensions
        # should probably re-export the models instead...
        model_payload = model_path
        model_id_lower = model_id_name.lower()
        
        if provider_str == "OpenVINOExecutionProvider":
            if "sam2" in model_id_lower:
                try:
                    print(f"[ModelManager] Applying OpenVINO stability fix for {model_id_name}...")
                    onnx_model = onnx.load(model_path)
                    for input_node in onnx_model.graph.input:
                        shape = input_node.type.tensor_type.shape
                        dims = shape.dim
                        if len(dims) > 0:
                            # Handle SAM2 dynamic points [?, ?, 2] etc.
                            if dims[0].HasField("dim_param") or dims[0].dim_value <= 0:
                                dims[0].dim_value = 1
                                dims[0].ClearField("dim_param")

                    model_payload = onnx_model.SerializeToString()
                except Exception as e:
                    print(f"[ModelManager] Warning: Stability fix failed: {e}")




        # These session options stop VRAM/RAM usage ballooning with subsequent model runs
        # by disabling automatic memory allocation
        # Doesn't seem to affect performance
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_pattern = False
        
        final_providers = []
        cache_dir = None
        
        def cache_dir_name(provider_str, provider_options, model_id_name):
            sub_dir_name = f"{provider_str}_{model_id_name}"
            if 'device_type' in provider_options:
                sub_dir_name = f"{provider_str}-{provider_options['device_type']}_{model_id_name}"
            sub_dir_name = "".join([c for c in sub_dir_name if c.isalnum() or c in "-_"])
            cache_dir = os.path.join(self.cache_root_dir, sub_dir_name)
            os.makedirs(cache_dir, exist_ok=True) 
            return cache_dir
        

        if provider_str == "TensorrtExecutionProvider":
            cache_dir = cache_dir_name(provider_str, provider_options, model_id_name)

            trt_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
            }
            trt_opts.update(provider_options) # Merge any other opts
            final_providers.append((provider_str, trt_opts))

        elif provider_str == "OpenVINOExecutionProvider":

            cache_dir = cache_dir_name(provider_str, provider_options, model_id_name)

            # Inject Cache Config
            ov_opts = {
                "cache_dir": cache_dir,
                "num_streams": 1, # Often helps stability
                #"precision": "FP32", 
            }
            ov_opts.update(provider_options) # Adds 'device_type': 'GPU' etc.
            final_providers.append((provider_str, ov_opts))

        else:

            final_providers.append((provider_str, provider_options))

        # Always add CPU as fallback
        if provider_str != "CPUExecutionProvider":
            final_providers.append("CPUExecutionProvider")

        return ort.InferenceSession(model_payload, sess_options=sess_options, providers=final_providers)
    

    # Automatic Models

    def clear_auto_cache(self):
        if self.loaded_automatic_models:
            for key in list(self.loaded_automatic_models.keys()):
                del self.loaded_automatic_models[key]
            self.loaded_automatic_models.clear()

        # grouping matting into automatic models for simplicity
        if self.loaded_matting_models:
            self.loaded_matting_models.clear()

        gc.collect()

    def get_auto_session(self, model_name, provider_data):
        prov_str, prov_opts, prov_code = provider_data
        model_path = os.path.join(self.model_root_dir, model_name + ".onnx")
        cache_key = f"{model_name}_{prov_code}"
        
        session = None
        load_time = 0.0

        if self.auto_cache_mode > 0 and cache_key in self.loaded_automatic_models:
            return self.loaded_automatic_models[cache_key], 0.0

        t_start = timer()

        # clear models before loading a new one if keep last or no caching selected
        if self.auto_cache_mode < 2:
            self.clear_auto_cache()
            
        session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
        load_time = (timer() - t_start) * 1000
        
        if self.auto_cache_mode > 0:
            self.loaded_automatic_models[cache_key] = session
            
        return session, load_time
    
    def run_auto_inference(self, session, image_crop, model_name, load_time, prov_code):
        """
        Runs inference on a PIL image using the provided session.
        Returns the mask as a numpy array (0.0 to 1.0)
        """

        # Some models can be exported with differing input dimensions
        # Try read these first, and use hardcoded values below otherwise
        input_shape = session.get_inputs()[0].shape
        target_h, target_w = input_shape[2], input_shape[3]

        if "modnet" in model_name.lower():
            # input size of 512 longest edge, multiple of 32 for other
            orig_w, orig_h = image_crop.width, image_crop.height
            
            if orig_w >= orig_h:
                target_w = 512
                target_h = int(512 * (orig_h / orig_w))
            else:
                target_h = 512
                target_w = int(512 * (orig_w / orig_h))

            target_w = target_w - (target_w % 32)
            target_h = target_h - (target_h % 32)
            
            target_w = max(target_w, 32)
            target_h = max(target_h, 32)

            # -1 to 1 normalization
            mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        elif "rmbg" in model_name.lower() and "2" in model_name.lower():
            target_h, target_w = 1024, 1024
            mean, std = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
        elif "isnet" in model_name or "rmbg" in model_name:
            mean, std = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
        else:
            # BEN2 and others use imagenet normalisation
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        img_r = image_crop.convert("RGB").resize((target_w, target_h), Image.BICUBIC)
        im = np.array(img_r) / 255.0
        tmp = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.float32)
        tmp[:, :, 0] = (im[:, :, 0] - mean[0]) / std[0]
        tmp[:, :, 1] = (im[:, :, 1] - mean[1]) / std[1]
        tmp[:, :, 2] = (im[:, :, 2] - mean[2]) / std[2]

        inp = np.expand_dims(tmp.transpose((2, 0, 1)), 0).astype(np.float32)

        t_start = timer()
        result = session.run(None, {session.get_inputs()[0].name: inp})[0]
        inference_time = (timer() - t_start) * 1000

        load_str = f"{load_time:.0f}ms" if load_time > 0 else "Cached"
        status = f"{model_name} ({prov_code.upper()}): Load {load_str} | Inf {inference_time:.0f}ms"

        mask = result[0][0]
        if "BiRefNet" in model_name or "mvanet" in model_name:
            mask = 1 / (1 + np.exp(-mask))

        denom = (mask.max() - mask.min()) or 1.0
        mask = (mask - mask.min()) / denom

        # ensure float32, otherwise opencv will fail if float64
        mask = mask.astype(np.float32)
        
        mask = cv2.resize(mask, (image_crop.width, image_crop.height), interpolation=cv2.INTER_LINEAR)
        final_mask = (mask * 255).astype(np.uint8)
        
        return final_mask, status
    

    def run_auto_inference_2step(self, session, image_pil, model_name, load_time, prov_code, padding_percent=0.15):
        """
        Chains two inference passes. 
        Pass 1: Entire image to find the subject.
        Pass 2: Zoomed crop of the subject for higher detail.
        """
        # Full imag epass
        mask_full, status_p1 = self.run_auto_inference(session, image_pil, model_name, load_time, prov_code)
        
        # Find Bounding Box of the first mask, filtering out nearly transparent pixels
        coords = np.column_stack(np.where(mask_full > 50))
        
        if coords.size == 0:
            return mask_full, f"{status_p1} (No subject detected in Pass 1)"

        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        
        orig_w, orig_h = image_pil.size
        
        # Add padding to give the model context
        w_pad = int((x2 - x1) * padding_percent)
        h_pad = int((y2 - y1) * padding_percent)
        
        x1 = max(0, x1 - w_pad)
        y1 = max(0, y1 - h_pad)
        x2 = min(orig_w, x2 + w_pad)
        y2 = min(orig_h, y2 + h_pad)
        
        # Pass 2
        crop_pil = image_pil.crop((x1, y1, x2, y2))
        
        mask_crop_arr, status_p2 = self.run_auto_inference(session, crop_pil, model_name, load_time, prov_code)
        
        final_mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
        
        final_mask_np[y1:y2, x1:x2] = mask_crop_arr
        
        combined_status = f"{status_p2} [2-Step Refined]"
        return final_mask_np, combined_status
    
    # --- SAM Models ---

    def clear_sam_cache(self, clear_loaded_models=True):
        self.sam_encoder = None
        self.sam_decoder = None
        self.sam_model_path = None
        self.encoder_output = None
        if clear_loaded_models==True:
            self.loaded_sam_models.clear()
            gc.collect()

    def init_sam_session(self, model_name, provider_data):
        model_path = os.path.join(self.model_root_dir, model_name)
        
        # Check active session
        if self.sam_model_path == model_path and self.sam_encoder:
            return True, " | Model Load: Cached"

        prov_str, prov_opts, prov_code = provider_data
        cache_key = f"{model_name}_{prov_code}"

        if self.sam_cache_mode > 0 and cache_key in self.loaded_sam_models:
            self.sam_encoder, self.sam_decoder = self.loaded_sam_models[cache_key]
            self.sam_model_path = model_path
            self.encoder_output = None
            return True, " | Model Load: Cached"

        try:
            s = timer()
            enc = self._create_inference_session(model_path + ".encoder.onnx", prov_str, prov_opts, model_name)
            dec = self._create_inference_session(model_path + ".decoder.onnx", prov_str, prov_opts, model_name)
            
            if self.sam_cache_mode == 1: self.loaded_sam_models.clear()
            if self.sam_cache_mode > 0: self.loaded_sam_models[cache_key] = (enc, dec)
            
            self.sam_encoder = enc
            self.sam_decoder = dec
            self.sam_model_path = model_path
            self.encoder_output = None
            
            # TRT Warmup
            if prov_str == "TensorrtExecutionProvider" and cache_key not in self._sam_trt_warmed:
                self._warmup_sam(model_name)
                self._sam_trt_warmed.add(cache_key)

            load_time = int((timer() - s) * 1000)
            
            return True, f" | Model Load {load_time}ms"
        except Exception as e:
            print(f"SAM Load Error: {e}")
            return False, str(e)

    def _warmup_sam(self, model_name):
        # Dispatch to v1 or v2 warmup based on name
        if "sam2" in model_name: self._warmup_sam2_trt(SAM_TRT_WARMUP_POINTS)
        else: self._warmup_sam1_trt(SAM_TRT_WARMUP_POINTS)

    def _warmup_sam1_trt(self, max_points):
        """
        Warmup for classic SAM / mobile_sam path used by run_sam_inference().
        """
        input_size = (684, 1024)
        dummy_img = np.zeros((input_size[0], input_size[1], 3), dtype=np.float32)
        enc_inputs = {self.sam_encoder.get_inputs()[0].name: dummy_img}
        embedding = self.sam_encoder.run(None, enc_inputs)[0]
        
        onnx_coord = np.zeros((1, max_points + 1, 2), dtype=np.float32)
        onnx_label = np.zeros((1, max_points + 1), dtype=np.float32)
        onnx_label[0, :max_points] = 1.0; onnx_label[0, -1] = -1.0
        
        run_inputs = {
            "image_embeddings": embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((1,), dtype=np.float32),
            "orig_im_size": np.array(input_size, dtype=np.float32)
        }
        # Filter inputs in case model differs. Some don't resize after inference
        actual_inputs = {k: v for k, v in run_inputs.items() if k in [i.name for i in self.sam_decoder.get_inputs()]}
        self.sam_decoder.run(None, actual_inputs)

    def _warmup_sam2_trt(self, max_points):
        """
        Warmup for SAM2 path used by run_samv2_inference().
        """
        enc_inputs = self.sam_encoder.get_inputs()
        h, w = enc_inputs[0].shape[2:]
        dummy_img = np.zeros((h, w, 3), dtype=np.float32).transpose(2,0,1)[None, ...]
        res = self.sam_encoder.run(None, {enc_inputs[0].name: dummy_img})
        
        mask_in = np.zeros((1, 1, h // 4, w // 4), dtype=np.float32)
        inputs = {
            'image_embed': res[2], 'high_res_feats_0': res[0], 'high_res_feats_1': res[1],
            'mask_input': mask_in, 'has_mask_input': np.array([0], dtype=np.float32),
            'orig_im_size': np.array([h, w], dtype=np.int32),
            'point_coords': np.zeros((1, max_points, 2), dtype=np.float32),
            'point_labels': np.ones((1, max_points), dtype=np.float32)
        }
        actual_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in self.sam_decoder.get_inputs()]}
        self.sam_decoder.run(None, actual_inputs)

    def run_sam1(self, image_crop, valid_coords, valid_labels, current_crop_rect, prov_code):
        target_size, input_size = 1024, (684, 1024)
        img_np = np.array(image_crop.convert("RGB"))
        t_start = timer()
        
        # Encoder optimization
        if self.encoder_output is None or self.last_crop_rect != current_crop_rect:
            scale = min(input_size[1] / img_np.shape[1], input_size[0] / img_np.shape[0])
            mat = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
            cv_img = cv2.warpAffine(img_np, mat[:2], (input_size[1], input_size[0]), flags=cv2.INTER_LINEAR)
            
            inp = {self.sam_encoder.get_inputs()[0].name: cv_img.astype(np.float32)}
            self.encoder_output = self.sam_encoder.run(None, inp)
            
            self.last_crop_rect = current_crop_rect
            self.last_transform = mat
            self.last_orig_size = img_np.shape[:2]
        
        enc_time = (timer() - t_start) * 1000
        
        # Decoder
        t_start = timer()
        embedding = self.encoder_output[0]
        # Append padding point
        coords = np.concatenate([np.array(valid_coords), np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        labels = np.concatenate([np.array(valid_labels), np.array([-1])], axis=0)[None, :].astype(np.float32)
        
        # Transform coords
        coords_aug = np.concatenate([coords, np.ones((1, coords.shape[1], 1), dtype=np.float32)], axis=2)
        coords_trans = np.matmul(coords_aug, self.last_transform.T)[:, :, :2].astype(np.float32)

        dec_inputs = {
            "image_embeddings": embedding, "point_coords": coords_trans, "point_labels": labels,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array(input_size, dtype=np.float32),
        }
        masks = self.sam_decoder.run(None, dec_inputs)[0]
        
        # Post-process
        inv_mtx = np.linalg.inv(self.last_transform)
        mask_resized = cv2.warpAffine(masks[0, 0, :, :], inv_mtx[:2], (self.last_orig_size[1], self.last_orig_size[0]), flags=cv2.INTER_LINEAR)
        final_mask = (mask_resized > 0.0).astype(np.uint8) * 255
        
        # although we binarise the mask, we could keep it soft using a sigmoid function
        # which may work better with alpha matting.
        # for now, continue to output binary mask
        # scale_intensity = 2 # increase low confidence areas to reduce softness
        # probs = 1 / (1 + np.exp(-(mask_resized * scale_intensity)))
        # probs = (probs - 0.5) * 2        # Rescale so that the 0.5 boundary becomes 0.0
        # final_mask = (np.clip(probs, 0, 1) * 255).astype(np.uint8) # clip away <0 (background)


        dec_time = (timer() - t_start) * 1000

        enc_str = f"{enc_time:.0f}ms" if enc_time > 0.1 else "Cached"
        status = f"SAM ({prov_code.upper()}): Enc {enc_str} | Dec {dec_time:.0f}ms"

        return final_mask, status

    def run_sam2(self, image_crop, valid_coords, valid_labels, current_crop_rect, prov_code):
        orig_h, orig_w = image_crop.height, image_crop.width
        t_start = timer()
        
        if self.encoder_output is None or self.last_crop_rect != current_crop_rect:
            enc_inputs = self.sam_encoder.get_inputs()
            h, w = enc_inputs[0].shape[2:]
            
            img_rgb = np.array(image_crop.convert("RGB"))
            input_img = cv2.resize(img_rgb, (w, h))
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            input_img = (input_img / 255.0 - mean) / std
            input_tensor = input_img.transpose(2, 0, 1)[None, ...].astype(np.float32)
            
            self.encoder_output = self.sam_encoder.run(None, {enc_inputs[0].name: input_tensor})
            self.last_crop_rect = current_crop_rect
            self.last_enc_shape = (h, w)
        
        enc_time = (timer() - t_start) * 1000
        t_start = timer()
        
        high0, high1, embed = self.encoder_output
        h, w = self.last_enc_shape
        
        points = np.array(valid_coords, dtype=np.float32)[None, ...]
        points[..., 0] = points[..., 0] / orig_w * w
        points[..., 1] = points[..., 1] / orig_h * h
        labels = np.array(valid_labels, dtype=np.float32)[None, ...]
        
        # Some model exports don't resize the mask (e.g. vietdev)
        # so we collect valid inputs for the model we are using
        inputs = {
            'image_embed': embed, 'high_res_feats_0': high0, 'high_res_feats_1': high1,
            'point_coords': points, 'point_labels': labels,
            'mask_input': np.zeros((1, 1, h//4, w//4), dtype=np.float32),
            'has_mask_input': np.array([0], dtype=np.float32),
            'orig_im_size': np.array([orig_h, orig_w], dtype=np.int32)
        }
        actual_inputs = {k: v for k, v in inputs.items() if k in [i.name for i in self.sam_decoder.get_inputs()]}
        
        res = self.sam_decoder.run(None, actual_inputs)
        mask = res[0].squeeze()
        if mask.ndim == 2: best_mask = mask
        else:
             scores = res[1].squeeze()
             idx = 0 if scores.ndim == 0 else np.argmax(scores)
             best_mask = mask[idx]
             
        final_mask = cv2.resize(best_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (final_mask > 0.0).astype(np.uint8) * 255
        # see notes in run_sam1 if want to use a soft mask, not binary, for alpha matting
        
        dec_time = (timer() - t_start) * 1000

        enc_str = f"{enc_time:.0f}ms" if enc_time > 0.1 else "Cached"
        status = f"SAM ({prov_code.upper()}): Enc {enc_str} | Dec {dec_time:.0f}ms"

        return binary, status

    def get_matting_session(self, model_name, provider_data):
        # uses cache clearing setting from automatic models, unsure if worth making it's own specific option
        prov_str, prov_opts, prov_code = provider_data
        cache_key = f"{model_name}_{prov_code}"
        if cache_key in self.loaded_matting_models:
            return self.loaded_matting_models[cache_key]
        
        if self.auto_cache_mode == 1: # Keep Last
            self.loaded_matting_models.clear()
            gc.collect()
            
        model_path = os.path.join(self.model_root_dir, model_name + ".onnx")
        session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
        if self.auto_cache_mode > 0:
            self.loaded_matting_models[cache_key] = session
        return session
    
    def run_matting(self, algorithm_name, image_pil, trimap_np, provider_data, longest_edge_limit=1024, allow_upscaling=False, alpha=None):

        orig_w, orig_h = image_pil.size
        
        # Calculate Target Dimensions constrained to longest edge, without upscaling
        current_longest_edge = max(orig_w, orig_h)
        
        # If upscaling is allowed (testing flag to see if increases quality)
        if allow_upscaling:
            effective_limit = longest_edge_limit
        else:
            effective_limit = min(current_longest_edge, longest_edge_limit)
        
        if orig_w >= orig_h:
            target_w = effective_limit
            target_h = int(effective_limit * (orig_h / orig_w))
        else:
            target_h = effective_limit
            target_w = int(effective_limit * (orig_w / orig_h))

        # Force to nearest multiple of 32 (Requirement for ViTMatte/IndexNet, will make little difference to pymatting)
        target_w = max(32, (target_w // 32) * 32)
        target_h = max(32, (target_h // 32) * 32)
        target_size = (target_w, target_h)
        
        img_resized = np.array(image_pil.convert("RGB").resize(target_size, Image.BILINEAR))
        tri_resized = cv2.resize(trimap_np, target_size, interpolation=cv2.INTER_NEAREST)
        if alpha:
            alpha_resized = alpha.resize(target_size, Image.BILINEAR)
        

        name_lower = algorithm_name.lower()
        if "vitmatte" in name_lower:
            session = self.get_matting_session(algorithm_name, provider_data)
            return self._run_vitmatte_inference(session, img_resized, tri_resized, image_pil.size)
        
        elif "indexnet" in name_lower:
            session = self.get_matting_session(algorithm_name, provider_data)
            return self._run_indexnet_inference(session, img_resized, tri_resized, image_pil.size)
        
        elif "withoutbg_focus_1" in name_lower:
            session = self.get_matting_session(algorithm_name, provider_data)
            return self._run_withoutbg(session, img_resized, alpha_resized, image_pil.size)
        
        else:
            return self._run_pymatting(img_resized, tri_resized, image_pil.size)

    def _run_pymatting(self, img_resized, tri_resized, original_size):
        """
        Calculates the alpha matte using the PyMatting library.
        Returns the alpha matte as a PIL Image.
        """
        s = timer()
        img_normalised = img_resized / 255.0
        trimap_normalised = tri_resized / 255.0

        alpha = estimate_alpha_sm(img_normalised, trimap_normalised)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        if alpha.shape[::-1] != original_size:
            alpha = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)
            
        print(f"PyMatting Shared Matting | Size: {img_resized.shape[::-1]} | Time: {timer()-s:.1f}s")
        return Image.fromarray(alpha, mode="L")

    def _run_vitmatte_inference(self, session, img_resized, tri_resized, original_size):
        """
        Runs inference with a ViTMatte ONNX model.
        Returns the alpha matte as a PIL Image.
        """
        
        image_np = (img_resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        trimap_np = np.expand_dims(tri_resized.astype(np.float32) / 255.0, axis=-1)
        
        combined = np.concatenate((image_np, trimap_np), axis=2)
        combined = np.transpose(combined, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: combined})

        alpha = outputs[0][0][0]
        alpha_image = Image.fromarray((alpha * 255).astype(np.uint8), mode='L')
        return alpha_image.resize(original_size, Image.LANCZOS)
    
    def _run_indexnet_inference(self, session, img_resized, tri_resized, original_size):
        h, w = img_resized.shape[:2]
        input_data = np.zeros((1, 4, h, w), dtype=np.float32)
        input_data[0, 0:3, :, :] = img_resized.transpose(2, 0, 1)
        input_data[0, 3, :, :] = tri_resized
        input_data /= 255.0

        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})[0]

        alpha = result[0][0]
        
        alpha[tri_resized == 0] = 0
        alpha[tri_resized == 255] = 255
        
        alpha_resized = cv2.resize(alpha, original_size, interpolation=cv2.INTER_LINEAR)
        
        alpha_final = np.clip(alpha_resized * 255, 0, 255).astype(np.uint8)
        # Final resize and trimap constraint
        return Image.fromarray(alpha_final, mode="L")

    def _run_withoutbg(self, refiner_session, img, alpha_mask_pil, original_size):
        # included for testing, but not included in model downloader currently
        # requires a tweak to UI ideally as it takes alpha, not a trimap

        # Scale to [0, 1]
        rgb_array = img / 255.0

        alpha_array = np.array(alpha_mask_pil, dtype=np.float32) / 255.0

        # Ensure alpha is single channel
        if len(alpha_array.shape) == 3:
            alpha_array = alpha_array[:, :, 0]

        # Concatenate RGB + alpha to create 4-channel input
        rgba_array = np.concatenate(
            [
                rgb_array,
                np.expand_dims(alpha_array, axis=2),
            ],
            axis=2,
        )

        # Prepare for model: transpose to CHW format and add batch dimension
        input_tensor = np.transpose(rgba_array, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
       
        input_name = refiner_session.get_inputs()[0].name
        ort_inputs = {input_name: input_tensor}
        ort_outs = refiner_session.run(None, ort_inputs)
        alpha_output = ort_outs[0]

        alpha_output = alpha_output.squeeze(0)
        if len(alpha_output.shape) == 3:
            alpha_output = alpha_output[0]

        # Normalize to 0-255 range
        alpha_output = np.clip(alpha_output * 255.0, 0, 255).astype(np.uint8)

        refined_alpha = Image.fromarray(alpha_output, mode="L")

        return refined_alpha.resize(original_size, Image.BILINEAR)

    def estimate_foreground(self, image_pil, alpha_mask_pil, algorithm='ml', radius = 90):
        """Refines foreground colors to remove background halos."""
        img_rgb = np.array(image_pil.convert("RGB")) / 255.0
        alpha = np.array(alpha_mask_pil.convert("L")) / 255.0
        
      
        if algorithm == 'ml':
            s=timer()
            fg_rgb = estimate_foreground_ml(img_rgb, alpha)
            print("pymatting estimate_foreground_ml",timer()-s)

        elif algorithm == 'blur_fusion_2':
            fg_rgb = estimate_fg_blur_fusion(img_rgb, alpha, radius=radius, refine_radius=6, downscale=0.5)

        elif algorithm == 'blur_fusion_1':
            fg_rgb = estimate_fg_blur_fusion(img_rgb, alpha, radius=radius, refine_radius=None, downscale=0.5)

        elif algorithm == 'cf':
            fg_rgb = estimate_foreground_cf(img_rgb, alpha)
        else:
            print("Invalid foreground estimation model choice. Check your settings and try again")
            return None




        fg_rgb = np.clip(fg_rgb * 255, 0, 255).astype(np.uint8)
        
        # Return as RGBA PIL image
        combined = np.dstack((fg_rgb, (alpha * 255).astype(np.uint8)))
        return Image.fromarray(combined, "RGBA")
    



    # Inpainting

    def get_inpainting_session(self, model_name, provider_data):
        """
        Gets or loads the inpainting session. 
        Reuse auto_cache settings for simplicity.
        """
        prov_str, prov_opts, prov_code = provider_data
        model_path = os.path.join(self.model_root_dir, model_name + ".onnx")
        cache_key = f"{model_name}_{prov_code}"

        if self.auto_cache_mode > 0 and cache_key in self.loaded_automatic_models:
             return self.loaded_automatic_models[cache_key]
        
        if self.auto_cache_mode < 2:
            self.clear_auto_cache()

        session = self._create_inference_session(model_path, prov_str, prov_opts, model_name)
        
        if self.auto_cache_mode > 0:
            self.loaded_automatic_models[cache_key] = session
            
        return session

    def run_lama_inpainting(self, image_pil, mask_pil, provider_data):
        """
        Runs LaMa inpainting on the provided image and mask.
        Adapted from ailia example for ONNX Runtime.
        """
        model_name = "lama"
        session = self.get_inpainting_session(model_name, provider_data)

        # Input image: float32, 0.0 - 1.0
        image_np = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
        orig_h, orig_w = image_np.shape[:2]

        mask_np = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
        # Ensure binary mask
        mask_np = (mask_np > 0).astype(np.float32)

        # Preprocess
        def ceil_modulo(x, mod):
            if x % mod == 0: return x
            return (x // mod + 1) * mod

        def pad_img_to_modulo(img, mod):
            if len(img.shape) == 3:
                height, width, channels = img.shape
                out_height = ceil_modulo(height, mod)
                out_width = ceil_modulo(width, mod)
                # Pad Height and Width, leave Channels (0,0)
                return np.pad(img, ((0, out_height - height), (0, out_width - width), (0, 0)), mode='symmetric')
            else:
                height, width = img.shape
                out_height = ceil_modulo(height, mod)
                out_width = ceil_modulo(width, mod)
                return np.pad(img, ((0, out_height - height), (0, out_width - width)), mode='symmetric')

        pad_out_to_modulo = 32
        
        image_padded = pad_img_to_modulo(image_np, pad_out_to_modulo)
        mask_padded = pad_img_to_modulo(mask_np, pad_out_to_modulo)

        # Transpose HWC -> CHW and add batch dimension -> NCHW
        input_image = np.transpose(image_padded, (2, 0, 1))[None, ...]
        input_mask = mask_padded[None, None, ...]

        # Inference
        t_start = timer()
        inputs = {
            session.get_inputs()[0].name: input_image,
            session.get_inputs()[1].name: input_mask
        }
        result = session.run(None, inputs)[0]
        
        inference_time = (timer() - t_start) * 1000

        # Post Process
        result = result[0]
        result = np.transpose(result, (1, 2, 0)) # CHW -> HWC
        
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Crop back to original size to remove padding
        result = result[:orig_h, :orig_w, :]

        result_img = Image.fromarray(result, "RGB").convert("RGBA")
        
        status = f"LaMa Inpaint: {inference_time:.0f}ms"
        return result_img, status
    

    def run_deepfill_inpainting(self, image_pil, mask_pil, provider_data, model_name="deepfillv2"):
        """
        Runs DeepFillv2 Inpainting. Handles final squaring if the input patch 
        is rectangular due to image boundary clamping.
        """
        session = self.get_inpainting_session(model_name, provider_data)

        # Handle Square Padding (Safety check for patches hit by image edges)
        orig_w, orig_h = image_pil.size
        if orig_w == orig_h:
            square_img = image_pil.convert("RGB")
            square_mask = mask_pil.convert("L")
            side = orig_w
            offset_x, offset_y = 0, 0
        else:
            side = max(orig_w, orig_h)
            square_img = Image.new("RGB", (side, side), (0, 0, 0))
            square_mask = Image.new("L", (side, side), 0)
            offset_x = (side - orig_w) // 2
            offset_y = (side - orig_h) // 2
            square_img.paste(image_pil.convert("RGB"), (offset_x, offset_y))
            square_mask.paste(mask_pil.convert("L"), (offset_x, offset_y))

        # Determine model resolution
        target_res = (256, 256)
        if "512" in model_name:
            target_res = (512, 512)
        elif "1024" in model_name:
            target_res = (1024, 1024)
            
        img_resized = square_img.resize(target_res, Image.BILINEAR)
        mask_resized = square_mask.resize(target_res, Image.NEAREST)

        image_np = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        
        mask_np = np.array(mask_resized).astype(np.float32) / 255.0
        mask_np = (mask_np > 0).astype(np.float32)
        mask_4d = mask_np[None, None, ...] 

        # Normalisation: [-1, 1]
        img_norm = (image_np.astype(np.float32) - 127.5) / 127.5
        img_t = np.transpose(img_norm, (2, 0, 1))[None, ...] 

        # Apply Mask to Input
        img_t_masked = img_t * (1.0 - mask_4d)

        # Construct Input Tensor
        ones_t = np.ones_like(mask_4d)
        input_data = np.concatenate((img_t_masked, ones_t, mask_4d), axis=1).astype(np.float32)

        # Inference
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})
        output_tensor = result[1] if len(result) > 1 else result[0]
        
        # Post Process - Internal Blending
        refined_blended_t = output_tensor * mask_4d + img_t * (1.0 - mask_4d)
        output_np = np.transpose(refined_blended_t[0], (1, 2, 0)) 
        
        # Denormalise
        output_np = ((output_np + 1.0) / 2.0 * 255.0)
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)

        # Convert back to RGB and crop out any safety padding
        result_rgb = cv2.cvtColor(output_np, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb, "RGB")
        result_pil = result_pil.resize((side, side), Image.LANCZOS)
        
        final_patch = result_pil.crop((offset_x, offset_y, offset_x + orig_w, offset_y + orig_h))

        return final_patch.convert("RGBA"), 0





    def run_matting_tiled(self, algorithm_name, image_pil, trimap_np, provider_data, tile_size=512, overlap=128):
        """
        Runs matting at 1:1 scale using overlapping tiles and linear blending.
        """

        s = timer()
        w, h = image_pil.size
        # Use float32 for accumulation to handle blending weights
        acc_alpha = np.zeros((h, w), dtype=np.float32)
        acc_weight = np.zeros((h, w), dtype=np.float32)

        # Pre-calculate a 2D linear ramp weight map for the tile
        # 1.0 in the center, 0.0 at the extreme edges of the overlap
        ramp = np.linspace(0, 1, overlap)
        single_ramp = np.ones(tile_size + 2 * overlap, dtype=np.float32)
        single_ramp[:overlap] = ramp
        single_ramp[-overlap:] = ramp[::-1]
        
        # 2D weight mask (H, W)
        weight_tile_base = np.outer(single_ramp, single_ramp)

        steps_x = range(0, w, tile_size)
        steps_y = range(0, h, tile_size)

        for ty in steps_y:
            for tx in steps_x:
                
                # Define the Patch with Overlap
                py1 = max(0, ty - overlap)
                py2 = min(h, ty + tile_size + overlap)
                px1 = max(0, tx - overlap)
                px2 = min(w, tx + tile_size + overlap)

                # Skip tiles that are 100% Background (0) or 100% Foreground (255)
                tile_trimap = trimap_np[py1:py2, px1:px2]
                if not np.any(tile_trimap == 128):
                    # Just copy the trimap value (already definite)
                    acc_alpha[py1:py2, px1:px2] += tile_trimap.astype(np.float32)
                    acc_weight[py1:py2, px1:px2] += 1.0
                    print("Tile", ty, tx,"Skipped")
                    continue
                
                print("Tile", ty, tx)

                patch_img = image_pil.crop((px1, py1, px2, py2))
                
                # Inference, passing a high limit to ensure no resizing of the patch
                res_alpha_pil = self.run_matting(algorithm_name, patch_img, tile_trimap, 
                                                provider_data, longest_edge_limit=2048)
                res_alpha = np.array(res_alpha_pil).astype(np.float32)

                # Apply Weights for Blending
                # Crop weight mask if patch is smaller than standard (at image edges)
                ph, pw = res_alpha.shape
                
                # Use a slice of our pre-calculated weight tile
                # This ensures we don't fade out at the true image boundaries
                w_h, w_w = weight_tile_base.shape
                # If we are at the top edge, we don't want to fade the top
                curr_weight = weight_tile_base.copy()
                if ty == 0: curr_weight[:overlap, :] = 1.0
                if tx == 0: curr_weight[:, :overlap] = 1.0
                if ty + tile_size >= h: curr_weight[-(w_h - (h-ty+overlap)):, :] = 1.0
                if tx + tile_size >= w: curr_weight[:, -(w_w - (w-tx+overlap)):] = 1.0
                
                # Crop weight to match actual patch size
                curr_weight = curr_weight[:ph, :pw]

                # Accumulate
                acc_alpha[py1:py2, px1:px2] += (res_alpha * curr_weight)
                acc_weight[py1:py2, px1:px2] += curr_weight

        # Normalize and convert back to uint8
        final_alpha = acc_alpha / (acc_weight + 1e-8)

        print("Tiled Matting Time:", timer()-s)
        return Image.fromarray(np.clip(final_alpha, 0, 255).astype(np.uint8))
  