from PyQt6.QtGui import (QPixmap, QImage)
import numpy as np
import cv2
import math
from timeit import default_timer as timer
import os
from PIL import Image, ImageFilter



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

def generate_drop_shadow(mask_pil, opacity, blur_radius, offset_x, offset_y, shadow_downscale=0.25):
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
    if shadow_downscale != 1.0:
        small_w = max(1, int(w * shadow_downscale))
        small_h = max(1, int(h * shadow_downscale))
        m_small = cv2.resize(m_np, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        m_small = m_np

    rad = int(blur_radius  * shadow_downscale)
    if rad % 2 == 0: rad += 1
    ksize = (rad, rad)

    # GaussianBlur on the downscaled mask
    m_blur_small = cv2.stackBlur(m_small, ksize)
    m_blur_small = cv2.convertScaleAbs(m_blur_small, alpha=opacity / 255.0)
    
    # Scale back up to original resolution
    if shadow_downscale != 1.0:
        m_full = cv2.resize(m_blur_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        m_full = m_blur_small
    
    # The shadow is black (0,0,0), we only populate the Alpha channel
    shifted_alpha = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate slice boundaries for the offset translation
    src_y1, src_y2 = max(0, -offset_y), min(h, h - offset_y)
    src_x1, src_x2 = max(0, -offset_x), min(w, w - offset_x)
    dst_y1, dst_y2 = max(0, offset_y), min(h, h + offset_y)
    dst_x1, dst_x2 = max(0, offset_x), min(w, w + offset_x)

    if dst_y2 > dst_y1 and dst_x2 > dst_x1:
        shifted_alpha[dst_y1:dst_y2, dst_x1:dst_x2] = m_full[src_y1:src_y2, src_x1:src_x2]
    
    z = np.zeros((h, w), dtype=np.uint8)
    shadow_layer_np = cv2.merge([z, z, z, shifted_alpha])
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
    \/:*?"<>|
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


def get_current_crop_bbox(mask, drop_shadow, sl_s_x, sl_s_y, sl_s_r):
    """
    Calculates the bounding box of the current mask, including shadows if enabled.
    Returns (min_x, min_y, max_x, max_y) or None if the mask is empty.
    """
    if not mask:
        return None

    bbox = mask.getbbox()
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
    orig_w, orig_h = mask.size
    final_min_x = max(0, int(min_x))
    final_min_y = max(0, int(min_y))
    final_max_x = min(orig_w, int(max_x))
    final_max_y = min(orig_h, int(max_y))

    if final_max_x <= final_min_x or final_max_y <= final_min_y:
        return None

    return (final_min_x, final_min_y, final_max_x, final_max_y)




def guided_filter(guide, src, radius, eps):
    """
    Numpy/OpenCV Guided Filter.
    guide: (H, W, C) or (H, W) float32 [0, 1]
    src:   (H, W) float32 [0, 1]
    """
    if guide.dtype != np.float32: guide = guide.astype(np.float32)
    if src.dtype != np.float32: src = src.astype(np.float32)

    # Convert to grayscale for matting-style refinement
    if len(guide.shape) == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)

    mean_I = cv2.boxFilter(guide, -1, (radius, radius))
    mean_p = cv2.boxFilter(src, -1, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * src, -1, (radius, radius))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(guide * guide, -1, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))
    
    q = mean_a * guide + mean_b
    return np.clip(q, 0, 1)


def generate_trimap_from_mask(mask_pil, fg_erode_size, bg_erode_size):
    """
    Generates a three-tone trimap from a binary mask using erosion.
    Returns the trimap as a NumPy array (0=BG, 128=Unknown, 255=FG).
    """
    #mask_np = np.array(mask_pil)

    if isinstance(mask_pil, np.ndarray):
        mask_np = mask_pil
    else:
        mask_np = np.array(mask_pil)


    foreground_threshold = 240
    background_threshold = 10

    is_foreground = mask_np > foreground_threshold
    is_background = mask_np < background_threshold

    # Erode foreground
    if fg_erode_size > 0:
        fg_kernel = np.ones((fg_erode_size, fg_erode_size), np.uint8)
        is_foreground_eroded = cv2.erode(is_foreground.astype(np.uint8), fg_kernel, iterations=1)
    else:
        is_foreground_eroded = is_foreground

    # Erode background
    if bg_erode_size > 0:
        bg_kernel = np.ones((bg_erode_size, bg_erode_size), np.uint8)
        is_background_eroded = cv2.erode(is_background.astype(np.uint8), bg_kernel, iterations=1)
    else:
        is_background_eroded = is_background

    trimap = np.full(mask_np.shape, dtype=np.uint8, fill_value=128)
    trimap[is_foreground_eroded.astype(bool)] = 255
    trimap[is_background_eroded.astype(bool)] = 0

    return trimap


def clean_alpha(mask):
    """
    Filters out alpha <5 and > 250
    """
    alpha_np = np.array(mask)
    alpha_np[alpha_np > 250] = 255
    alpha_np[alpha_np < 5] = 0
    return Image.fromarray(alpha_np)

def generate_alpha_map(mask):
        """
        Creates a diagnostic image to reveal hidden transparency.
        - Black: Fully Transparent (0)
        - Red: Semi Transparent
        - White: Fully Opaque (255)
        """
        alpha_np = np.array(mask)
    
        # white default array
        h, w = alpha_np.shape
        viz_np = np.full((h, w, 3), 255, dtype=np.uint8)
        
        # pure transparent to black
        viz_np[alpha_np == 0] = [0, 0, 0]
        
        # semi to red
        semi_transparent_mask = (alpha_np > 0) & (alpha_np < 255)
        viz_np[semi_transparent_mask] = [255, 0, 0]
        
        return Image.fromarray(viz_np)


def generate_outline(mask_pil, size, color_tuple, threshold=128, opacity=255):
    """
    Generates a solid-colored, high-quality smooth outline using a 
    Distance Transform (SDF) approach.

    Args:
        mask_pil: PIL Image ('L' mode) of the mask.
        size: The thickness of the outline in pixels.
        color_tuple: An (R, G, B) tuple for the outline colour.
        threshold: The alpha value (0-255) at which the outline begins.
        opacity: The global opacity of the outline (0-255).
    Returns:
        A PIL Image ('RGBA' mode) of the smooth outline silhouette.
    """
    if size <= 0:
        return Image.new("RGBA", mask_pil.size, (0, 0, 0, 0))

    if isinstance(mask_pil, np.ndarray):
        mask_np = mask_pil
    else:
        mask_np = np.array(mask_pil)

    # Binarise using the user-provided threshold. 
    _, binary_mask = cv2.threshold(mask_np, threshold, 255, cv2.THRESH_BINARY)

    # Distance Transform
    inverted_mask = cv2.bitwise_not(binary_mask)
    dist_field = cv2.distanceTransform(inverted_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Threshold the distance field
    outline_mask = np.where(dist_field <= size, 255, 0).astype(np.uint8)

    # Final Geometric Smoothing
    smooth_rad = max(3, int(size * 0.3))
    if smooth_rad % 2 == 0: smooth_rad += 1
    
    blurred = cv2.GaussianBlur(outline_mask, (smooth_rad, smooth_rad), 0)
    _, final_mask = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)
    
    # Anti-Aliasing
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)

    # Assemble the RGBA layer
    h, w = mask_np.shape
    r, g, b = color_tuple
    color_layer = np.zeros((h, w, 4), dtype=np.uint8)
    color_layer[:, :, 0] = r
    color_layer[:, :, 1] = g
    color_layer[:, :, 2] = b
    
    # Apply user opacity to the final mask
    if opacity < 255:
        alpha_channel = (final_mask.astype(np.float32) * (opacity / 255.0)).astype(np.uint8)
    else:
        alpha_channel = final_mask

    color_layer[:, :, 3] = alpha_channel
    
    return Image.fromarray(color_layer, "RGBA")


def generate_inner_glow(mask_pil, size, color_tuple, threshold=128, opacity=255):
    """
    Generates an inner glow layer with thresholding to prevent over-lighting 
    faint edges.
    
    Args:
        mask_pil: PIL Image ('L' mode) of the mask.
        size: The thickness of the glow in pixels.
        color_tuple: An (R, G, B) tuple for the glow colour.
        threshold: The alpha value (0-255) to consider as the "solid" edge.
        opacity: Global opacity (0-255).
    """
    if size <= 0:
        return Image.new("RGBA", mask_pil.size, (0, 0, 0, 0))

    if isinstance(mask_pil, np.ndarray):
        mask_np = mask_pil
    else:
        mask_np = np.array(mask_pil)

    # Binarise to ensure the glow calculates distance from a clean edge, not faint noise.
    _, binary_mask = cv2.threshold(mask_np, threshold, 255, cv2.THRESH_BINARY)

    # Distance Transform
    dist_field = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Internal gradient
    glow_mask = np.clip(1.0 - (dist_field / size), 0, 1)
    
    # Constrain to the binarised area
    glow_mask[binary_mask == 0] = 0
    
    # Smoothing
    glow_mask = (glow_mask * 255).astype(np.uint8)
    smooth_rad = max(3, int(size * 0.5))
    if smooth_rad % 2 == 0: smooth_rad += 1
    glow_mask = cv2.GaussianBlur(glow_mask, (smooth_rad, smooth_rad), 0)

    # Assemble RGBA
    h, w = mask_np.shape
    r, g, b = color_tuple
    glow_layer = np.zeros((h, w, 4), dtype=np.uint8)
    glow_layer[:, :, 0] = r
    glow_layer[:, :, 1] = g
    glow_layer[:, :, 2] = b
    
    # Multiply glow by distance gradient (glow_mask), opacity and original mask
    # to ensure the glow softens with the hair.
    alpha_f = (glow_mask.astype(np.float32) / 255.0) * \
              (opacity / 255.0) * \
              (mask_np.astype(np.float32) / 255.0)
              
    glow_layer[:, :, 3] = (alpha_f * 255).astype(np.uint8)
    
    return Image.fromarray(glow_layer, "RGBA")


def apply_subject_tint(image_pil, color_tuple, amount):
    """
    Applies a color tint to an RGBA image.
    
    Args:
        image_pil: The subject cutout (PIL RGBA).
        color_tuple: (R, G, B) to tint with.
        amount: Intensity (0.0 to 1.0).
    """
    if amount <= 0:
        return image_pil

    img_np = np.array(image_pil).astype(np.float32)
    h, w, _ = img_np.shape
    
    # Create a target color buffer
    r, g, b = color_tuple
    tint_layer = np.full((h, w, 3), [r, g, b], dtype=np.float32)
    
    # Blend original RGB with tint RGB
    original_rgb = img_np[:, :, :3]
    blended_rgb = (original_rgb * (1.0 - amount)) + (tint_layer * amount)
    
    img_np[:, :, :3] = blended_rgb
    
    return Image.fromarray(img_np.astype(np.uint8), "RGBA")

def compose_final_image(original_image, mask, settings, model_manager=None,
                        cached_foreground=None, precomputed_bg=None):
    """
    Composes the final image based on the original image, mask, and settings.
    Supports injecting precomputed layers (foreground_layer, background) for caching in GUI.
    Otherwise generates layers for use in batch processing
    """
    
    current_mask = mask
    if settings.get("clean_alpha", True):
        current_mask = clean_alpha(current_mask)

    # 2. Prepare Cutout (Foreground)
    foreground_layer = cached_foreground
    
    if foreground_layer is None:
        fg_settings = settings.get("foreground_correction", {})
        if fg_settings.get("enabled", False) and model_manager:
            algo = fg_settings.get("algorithm", "ml")
            foreground_layer = model_manager.estimate_foreground(original_image, current_mask, algo)
        else:
            foreground_layer = original_image.convert("RGBA")
            foreground_layer.putalpha(current_mask)

    # Apply Effects to Cutout
    
    # Tint
    tint_settings = settings.get("tint", {})
    if tint_settings.get("enabled", False):
        foreground_layer = apply_subject_tint(
            foreground_layer,
            tint_settings.get("color"),
            tint_settings.get("amount")
        )

    # Outline
    outline_settings = settings.get("outline", {})
    if outline_settings.get("enabled", False):
        outline_layer = generate_outline(
            current_mask,
            size=outline_settings.get("size"),
            color_tuple=outline_settings.get("color"),
            threshold=outline_settings.get("threshold"),
            opacity=outline_settings.get("opacity")
        )
        foreground_layer = Image.alpha_composite(outline_layer, foreground_layer)

    # Drop Shadow
    shadow_settings = settings.get("shadow", {})
    if shadow_settings.get("enabled", False):
        shadow_layer = generate_drop_shadow(
            current_mask,
            opacity=shadow_settings.get("opacity"),
            blur_radius=shadow_settings.get("radius"),
            offset_x=shadow_settings.get("x"),
            offset_y=shadow_settings.get("y"),
            shadow_downscale=shadow_settings.get("downscale", 0.125)
        )
        foreground_layer = Image.alpha_composite(shadow_layer, foreground_layer)

    # Compose with Background
    bg_settings = settings.get("background", {})
    bg_type = bg_settings.get("type", "Transparent")
    
    final = None
    
    if bg_type == "Transparent":
        final = foreground_layer
    elif bg_type == "Original Image":
        final = original_image.convert("RGBA")
        final.alpha_composite(foreground_layer)
    elif "Blur" in bg_type:
        if precomputed_bg:
            final = precomputed_bg.copy()
        else:
            final = generate_blurred_background(
                original_image, 
                current_mask, 
                bg_settings.get("blur_radius", 30)
            )
        final.alpha_composite(foreground_layer)
    else:
        # Solid Color
        color = bg_settings.get("color", (0, 0, 0)) # Default black
        # If color is a string (e.g. from known colors), handle or assume tuple
        if isinstance(color, str):
             final = Image.new("RGBA", original_image.size, color.lower())
        else:
             final = Image.new("RGBA", original_image.size, color)
             
        final.alpha_composite(foreground_layer)

    # Inner Glow
    ig_settings = settings.get("inner_glow", {})
    if ig_settings.get("enabled", False):
        glow_layer = generate_inner_glow(
            current_mask, 
            ig_settings.get("size"),
            ig_settings.get("color"),
            ig_settings.get("threshold"),
            ig_settings.get("opacity")
        )
        final = Image.alpha_composite(final, glow_layer)

    return final

def refine_mask(base_mask, image, settings, model_manager, trimap_np=None):
    """
    Refines a mask using alpha matting, softening, or binarisation.
    """
    processed_mask = base_mask

    # Alpha Matting
    matting_settings = settings.get("matting", {})
    if matting_settings.get("enabled", False) and model_manager:
        # Prepare params
        limit = matting_settings.get("longest_edge_limit", 1024)
        fg_erode = matting_settings.get("fg_erode", 15)
        bg_erode = matting_settings.get("bg_erode", 15)
        algo = matting_settings.get("algorithm", "vitmatte")
        provider = matting_settings.get("provider_data", None)

        # Generate Trimap (Batch mode always uses auto trimap for now)
        if trimap_np is None:
            trimap_np = generate_trimap_from_mask(processed_mask, fg_erode, bg_erode)
        
        matted_alpha = model_manager.run_matting(
            algo,
            image,
            trimap_np,
            provider,
            longest_edge_limit=limit
        )
        if matted_alpha:
            processed_mask = matted_alpha

    # Soften
    if settings.get("soften", False):
        processed_mask = processed_mask.filter(ImageFilter.GaussianBlur(radius=settings.get("soften_radius", 1.5)))

    # Binarise
    if settings.get("binarise", False):
        arr = np.array(processed_mask) > 128
        kernel = np.ones((3,3), np.uint8)
        arr = cv2.erode(cv2.dilate(arr.astype(np.uint8), kernel, iterations=1), kernel, iterations=1).astype(bool)
        processed_mask = Image.fromarray((arr*255).astype(np.uint8))
        
    return processed_mask
