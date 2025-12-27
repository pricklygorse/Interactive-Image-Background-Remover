from PyQt6.QtGui import (QPixmap, QImage)
import numpy as np
import cv2
import math
from timeit import default_timer as timer



def pil2pixmap(im):
    if im is None: return QPixmap()
    
    if im.mode == "RGB":
        im = im.convert("RGBA")
    elif im.mode == "L":
        im = im.convert("RGBA")

    if isinstance(im, np.ndarray):
        img_data = im
        height, width, channels = img_data.shape
    else:
        img_data = np.array(im, copy=False)
        width, height = im.size
        
    bytes_per_line = width * 4 
    
    qim = QImage(img_data.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)

    return QPixmap.fromImage(qim)

def numpy_to_pixmap(np_array):
    if np_array is None:
        return QPixmap()
    
    if np_array.dtype != np.uint8:
        if np_array.max() > 1.0:
            np_array = (np_array / 255.0)
        np_array = (np_array * 255).astype(np.uint8)

    h, w, ch = np_array.shape
    bytes_per_line = ch * w
    
    q_image = QImage(np_array.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
    
    return QPixmap.fromImage(q_image.copy())

def numpy_bgra_to_pixmap(np_array):
    """
    Converts a BGRA NumPy array to a QPixmap
    Expects (H, W, 4) uint8 BGRA.
    """
    height, width, channel = np_array.shape
    bytes_per_line = width * 4
    
    q_img = QImage(
        np_array.data, 
        width, 
        height, 
        bytes_per_line, 
        QImage.Format.Format_ARGB32
    )
    
    return QPixmap.fromImage(q_img)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#@profile
def apply_tone_sharpness(input_img_np, params):
    """
    Applies all current filters to the given input_img
    Expects (H, W, 4) BGRA.
    Returns (H, W, 4) BGRA.
    """
    s = timer()
    img_np = input_img_np
    
    # 1. Apply LUT (Includes Alpha identity if 4ch)
    lut = calculate_global_lut(params)
    img_np = cv2.LUT(img_np, lut)
    
    # 2. Saturation
    sat = params.get('saturation', 100) / 100.0
    if abs(sat - 1.0) > 0.01:
        # faster liner interpolation instead of the slower but more colour accurate saturation matrix
        # matrix = calculate_saturation_matrix(sat)
        # img_np = cv2.transform(img_np, matrix)
        current_alpha = img_np[:, :, 3]
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
        gray_bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
        img_np = cv2.addWeighted(img_np, sat, gray_bgra, 1.0 - sat, 0)
        img_np[:, :, 3] = current_alpha

    # 3. Unsharp Mask
    amt = params.get('unsharp_amount', 0)
    if amt > 0.5:
        radius = params.get('unsharp_radius', 1)
        threshold_val = params.get('unsharp_threshold', 0)
        
        k = int(radius * 2)
        if k % 2 == 0: k += 1
        k = max(1, k)
        
        # stackBlur handles 3 or 4 channels natively
        blurred = cv2.stackBlur(img_np, (k, k))
        
        a_factor = amt / 100.0
        
        if threshold_val <= 0:
            cv2.addWeighted(img_np, 1.0 + a_factor, blurred, -a_factor, 0, dst=img_np)
        else:
            sharpened = cv2.addWeighted(img_np, 1.0 + a_factor, blurred, -a_factor, 0)
            
            diff = cv2.absdiff(img_np, blurred)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGRA2GRAY)
            _, mask = cv2.threshold(diff_gray, threshold_val, 255, cv2.THRESH_BINARY)
            
            cv2.copyTo(sharpened, mask, img_np)

    print(timer()-s)
    return img_np

def calculate_global_lut(params):
    # Base indices [0-255]
    x = np.arange(256, dtype=np.float32)
    
    # White Balance Gains
    temp = params.get('white_balance', 6500)
    t = temp / 100
    if t <= 66:
        r_g, g_g, b_g = 255, 99.47 * math.log(t) - 161.1, (0 if t <= 19 else 138.5 * math.log(t - 10) - 305)
    else:
        r_g, g_g, b_g = 329.7 * math.pow(t - 60, -0.133), 288.1 * math.pow(t - 60, -0.075), 255
    
    r_scale, g_scale, b_scale = 255/max(1, r_g), 255/max(1, g_g), 255/max(1, b_g)
    
    # Tone Curve (Scaled /100.0)
    tone = params.get('tone_curve', 10) / 100.0
    
    hl_mask = 1 / (1 + np.exp(-(x - 192) / (255 * tone)))
    sh_mask = 1 - (1 / (1 + np.exp(-(x - 64) / (255 * tone))))
    mt_mask = 1 - hl_mask - sh_mask
    
    # Curve Factors (Scaled /100.0)
    curve_factors = ( (params.get('highlight', 100) / 100.0) * hl_mask + 
                        (params.get('midtone', 100) / 100.0) * mt_mask + 
                        (params.get('shadow', 100) / 100.0) * sh_mask )

    # Brightness & Contrast (Scaled /100.0)
    bright = params.get('brightness', 100) / 100.0
    contrast = params.get('contrast', 100) / 100.0
    
    def apply_all(val_range, scale):
        v = val_range * scale 
        v = v * curve_factors 
        v = v * bright        
        v = (v - 128) * contrast + 128
        return np.clip(v, 0, 255).astype(np.uint8)

    # outputs BGR for OpenCV
    lut_r = apply_all(x, r_scale)
    lut_g = apply_all(x, g_scale)
    lut_b = apply_all(x, b_scale)
    
    # Identity LUT for Alpha channel (no change to transparency)
    lut_a = np.arange(256, dtype=np.uint8)

    # Changed: Stack 4 channels instead of 3
    return np.stack([lut_b, lut_g, lut_r, lut_a], axis=-1).reshape(256, 1, 4)

def calculate_saturation_matrix(sat):
    """Generates a 4x4 matrix for saturation adjustment in BGRA space."""
    # Standard Luma coefficients for Rec. 709
    wr, wg, wb = 0.2126, 0.7152, 0.0722
    
    # Matrix components
    inv_sat = 1.0 - sat
    r_lum, g_lum, b_lum = wr * inv_sat, wg * inv_sat, wb * inv_sat
    
    # BGRA order matrix (4x4)
    # The last row/col is the Identity [0,0,0,1] for Alpha preservation
    matrix = np.array([
        [b_lum + sat, g_lum,       r_lum,       0.0],
        [b_lum,       g_lum + sat, r_lum,       0.0],
        [b_lum,       g_lum,       r_lum + sat, 0.0],
        [0.0,         0.0,         0.0,         1.0]
    ], dtype=np.float32)
    
    return matrix