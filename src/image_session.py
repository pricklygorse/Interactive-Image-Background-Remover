import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageGrab, ImageChops

from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt

from src.constants import UNDO_STEPS

class ImageSession:
    """
    Encapsulates a single image session, handling loading from file or clipboard,
    initial mask extraction from transparency, and maintaining all per-image state.
    """
    def __init__(self, source):
        self.source = source
        self.filename = "Clipboard" if source == "Clipboard" else os.path.basename(source)
        self.raw_original_image = None
        self.working_orig_image = None
        self.initial_mask = None
        self.adjustment_source_np = None
        self.image_exif = None
        self.size = (0, 0)
        
        # Comprehensive state variables
        self.working_mask = None
        self.model_output_mask = None
        self.refined_preview_mask = None
        self.undo_history = []
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        # Caching and refinement state
        self.working_mask_hash = None
        self.cached_fg_corrected_cutout = None
        self.cached_blurred_bg = None
        self.last_blur_params = None
        self.last_trimap = None
        self.user_trimap = None
        self.paint_image = None
        
    def load(self):
        """Loads the image from the specified source (file path, 'Clipboard', or 'None' for blank)."""
        if self.source == "Clipboard":
            img = ImageGrab.grabclipboard()
            if not isinstance(img, Image.Image):
                raise ValueError("No image found on the clipboard.")
        elif self.source == "None":
            # Create a blank transparent image
            img = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
        else:
            if not os.path.exists(self.source):
                raise FileNotFoundError(f"File not found: {self.source}")
            img = Image.open(self.source)
        
        # Apply EXIF transposition (handle rotation)
        if self.source != "None":
            img = ImageOps.exif_transpose(img)
            self.image_exif = img.info.get('exif')
        
        # Extract initial mask if there's transparency
        self.initial_mask = self._extract_initial_mask(img)
        
        # Convert to RGBA for internal processing
        self.raw_original_image = img.convert("RGBA")
        self.working_orig_image = self.raw_original_image.copy()
        self.size = self.raw_original_image.size
        
        # Create NumPy buffer for the adjustment pipeline (BGRA for OpenCV compatibility)
        self.adjustment_source_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.raw_original_image), cv2.COLOR_RGBA2BGRA)
        )
        
        # Initialize remaining buffers
        self.init_session_buffers(self.initial_mask)
        
        return self

    def init_session_buffers(self, initial_mask=None):
        """Initializes or resets masks, history, and scratchpads for this session."""
        if initial_mask:
            self.working_mask = initial_mask.convert("L").copy()
        else:
            self.working_mask = Image.new("L", self.size, 0)
            
        self.model_output_mask = None
        self.refined_preview_mask = None
        self.undo_history = [self.working_mask.copy()]
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        self.working_mask_hash = None
        self.cached_fg_corrected_cutout = None
        self.cached_blurred_bg = None
        self.last_blur_params = None
        self.last_trimap = None
        self.user_trimap = None
            
        # Scratchpad for painting
        self.paint_image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)

    def _extract_initial_mask(self, img):
        """Extracts the alpha channel as an initial mask if it contains transparency."""
        has_alpha = 'A' in img.mode or (img.mode == 'P' and 'transparency' in img.info)
        if has_alpha:
            alpha = img.getchannel('A')
            extrema = alpha.getextrema()
            # If min is 255, the alpha is solid white (no transparency)
            if extrema and extrema[0] < 255:
                return alpha
        return None

    def add_undo_step(self):
        """Caches the current mask state onto the undo history stack."""
        if self.working_mask:
            self.undo_history.append(self.working_mask.copy())
            if len(self.undo_history) > UNDO_STEPS:
                self.undo_history.pop(0)
            self.redo_history.clear()

    def undo(self):
        """Reverts the mask to the previous state in history."""
        if len(self.undo_history) > 0:
            self.redo_history.append(self.working_mask.copy())
            self.working_mask = self.undo_history.pop()
            return True
        return False

    def redo(self):
        """Restores the next mask state from the redo history stack."""
        if len(self.redo_history) > 0:
            self.undo_history.append(self.working_mask.copy())
            if len(self.undo_history) > UNDO_STEPS:
                self.undo_history.pop(0)
            self.working_mask = self.redo_history.pop()
            return True
        return False

    def is_mask_modified(self):
        """Checks if the working mask has data or history."""
        if len(self.undo_history) > 1:
            return True
        if self.working_mask:
            # Check for non-zero pixels
            extrema = self.working_mask.getextrema()
            if extrema and extrema[1] > 0:
                return True
        return False

    def reset_working_mask(self):
        """Wipes the current mask and history."""
        self.add_undo_step()
        self.working_mask = Image.new("L", self.size, 0)
        self.model_output_mask = None
        self.refined_preview_mask = None
        self.last_trimap = None
        self.user_trimap = None
        self.sam_coordinates = []
        self.sam_labels = []

    def copy_input_to_output(self):
        """Copies the alpha channel of the original image into the working mask."""
        if self.raw_original_image:
            self.add_undo_step()
            self.working_mask = self.raw_original_image.split()[3]
            return True
        return False

    def rotate(self, pil_transpose, cv2_rotate_code):
        """Rotates the entire session's per-image state."""
        self.raw_original_image = self.raw_original_image.transpose(pil_transpose)
        self.working_orig_image = self.working_orig_image.transpose(pil_transpose)
        
        if self.adjustment_source_np is not None:
            self.adjustment_source_np = cv2.rotate(self.adjustment_source_np, cv2_rotate_code)
            
        if self.working_mask:
            self.working_mask = self.working_mask.transpose(pil_transpose)
        if self.model_output_mask:
            self.model_output_mask = self.model_output_mask.transpose(pil_transpose)
        if self.user_trimap:
            self.user_trimap = self.user_trimap.transpose(pil_transpose)
            self.last_trimap = np.array(self.user_trimap)
            
        self.size = self.raw_original_image.size
        # Clear state that is no longer valid for the new orientation
        # undo could probably be implemented
        self.undo_history = [self.working_mask.copy()]
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        # Re-init paint scratchpad
        self.paint_image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)

    def crop(self, box):
        """Crops the session state to the given bounding box."""
        self.raw_original_image = self.raw_original_image.crop(box)
        self.working_orig_image = self.working_orig_image.crop(box)
        
        # Update the NumPy buffer from the cropped raw image, so no adjustments are baked in
        self.adjustment_source_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.raw_original_image), cv2.COLOR_RGBA2BGRA)
        )
        
        if self.working_mask:
            self.working_mask = self.working_mask.crop(box)
        if self.model_output_mask:
            self.model_output_mask = self.model_output_mask.crop(box)
        if self.user_trimap:
            self.user_trimap = self.user_trimap.crop(box)
            self.last_trimap = np.array(self.user_trimap)
            
        self.size = self.working_orig_image.size

        # Since dimensions changed, old undo history is invalid
        # Possibly could be implemented properly
        self.undo_history = [self.working_mask.copy()]
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        # Re-initialise the paint scratchpad
        self.paint_image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)


