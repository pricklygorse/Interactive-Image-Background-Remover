from PyQt6.QtGui import (QPixmap, QImage)
import numpy as np
import cv2
import math
from timeit import default_timer as timer
import os
from PIL import Image




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
    #s = timer()
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

    #print(timer()-s)
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


def generate_drop_shadow(mask_pil, opacity, blur_radius, offset_x, offset_y, shadow_downscale=0.125):
    """
    Generates a shadow layer based on a mask. 
    Optimised using numpy and downscaling for performance
    """
    w, h = mask_pil.size
    if isinstance(mask_pil, np.ndarray):
        m_np = mask_pil
    else:
        m_np = np.array(mask_pil)

    # Scale down for fast blur processing
    small_w = max(1, int(w * shadow_downscale))
    small_h = max(1, int(h * shadow_downscale))
    
    m_small = cv2.resize(m_np, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    #blur_size = max(1, int(blur_radius * shadow_downscale))

    rad = int(blur_radius  * shadow_downscale)
    if rad % 2 == 0: rad += 1
    ksize = (rad, rad)

    # GaussianBlur on the downscaled mask
    m_blur_small = cv2.stackBlur(m_small, ksize)
    m_blur_small = cv2.convertScaleAbs(m_blur_small, alpha=opacity / 255.0)
    
    # Scale back up to original resolution
    m_full = cv2.resize(m_blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create the RGBA shadow layer
    shadow_layer_np = np.zeros((h, w, 4), dtype=np.uint8)
    # The shadow is black (0,0,0), we only populate the Alpha channel
    shifted_alpha = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate slice boundaries for the offset translation
    src_y1, src_y2 = max(0, -offset_y), min(h, h - offset_y)
    src_x1, src_x2 = max(0, -offset_x), min(w, w - offset_x)
    dst_y1, dst_y2 = max(0, offset_y), min(h, h + offset_y)
    dst_x1, dst_x2 = max(0, offset_x), min(w, w + offset_x)

    if dst_y2 > dst_y1 and dst_x2 > dst_x1:
        shifted_alpha[dst_y1:dst_y2, dst_x1:dst_x2] = m_full[src_y1:src_y2, src_x1:src_x2]
    
    shadow_layer_np[:, :, 3] = shifted_alpha
    return Image.fromarray(shadow_layer_np)


def generate_blurred_background(image, mask, blur_radius):
    """
    Uses a weighted/normalised convolution technique to blur the background 
    without the subject's colours bleeding into the blur.
    """
    if isinstance(image, np.ndarray):
        orig_np = image
    else:
        orig_np = np.array(image)

    rgb = orig_np[:, :, :3]
    
    if isinstance(mask, np.ndarray):
        m_np = mask
    else:
        m_np = np.array(mask)

    # Expand mask slightly to ensure the subject edges are fully excluded from the blur source
    dilation_size = 7 
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_mask = cv2.dilate(m_np, kernel, iterations=1)
    
    # Create weight map (Background = 1.0, Dilated Subject = 0.0)
    weight_map = (255 - dilated_mask).astype(np.float32) / 255.0
              
    rad = blur_radius
    if rad % 2 == 0: rad += 1
    ksize = (rad, rad)

    # Apply weighted stackBlur (Normalised Convolution)
    # stackBlur for nearly gaussian blur at much faster speed
    weighted_blur = cv2.stackBlur(rgb * weight_map[..., None], ksize)
    blurred_weights = cv2.stackBlur(weight_map, ksize)

    # Divide by blurred weights to normalise intensity
    result = weighted_blur / (blurred_weights[..., None] + 1e-8)
    
    blur_final_np = cv2.convertScaleAbs(result)
    return Image.fromarray(blur_final_np).convert("RGBA")


def sanitise_filename_for_windows(path: str) -> str:

    """
    On Windows, strip invalid filename characters from the basename:
    \\/:*?"<>|
    Directory part (e.g. C:\\folder) is left intact.
    """
    if os.name != "nt":
        return path

    directory, basename = os.path.split(path)
    invalid_chars = r'\/:*?"<>|'
    cleaned = ''.join(c for c in basename if c not in invalid_chars)

    if not cleaned:
        cleaned = "output"

    return os.path.join(directory, cleaned)


def get_current_crop_bbox(working_mask, drop_shadow, sl_s_x, sl_s_y, sl_s_r):
    """
    Calculates the bounding box of the current mask, including shadows if enabled.
    Returns (min_x, min_y, max_x, max_y) or None if the mask is empty.
    """
    if not working_mask:
        return None

    bbox = working_mask.getbbox()
    if not bbox:
        return None

    min_x, min_y, max_x, max_y = bbox

    if drop_shadow:
        shadow_off_x = sl_s_x
        shadow_off_y = sl_s_y
        s_rad = sl_s_r

        # Expand bounding box to include the shadow and its blur radius
        s_min_x = min_x + shadow_off_x - s_rad
        s_min_y = min_y + shadow_off_y - s_rad
        s_max_x = max_x + shadow_off_x + s_rad
        s_max_y = max_y + shadow_off_y + s_rad

        min_x = min(min_x, s_min_x)
        min_y = min(min_y, s_min_y)
        max_x = max(max_x, s_max_x)
        max_y = max(max_y, s_max_y)

    # Clamp to image boundaries
    orig_w, orig_h = working_mask.size
    final_min_x = max(0, int(min_x))
    final_min_y = max(0, int(min_y))
    final_max_x = min(orig_w, int(max_x))
    final_max_y = min(orig_h, int(max_y))

    if final_max_x <= final_min_x or final_max_y <= final_min_y:
        return None

    return (final_min_x, final_min_y, final_max_x, final_max_y)