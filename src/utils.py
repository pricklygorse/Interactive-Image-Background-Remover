from PyQt6.QtGui import (QPixmap, QImage)
import numpy as np


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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))