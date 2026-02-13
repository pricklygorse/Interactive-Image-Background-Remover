import os

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageGrab
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage

from src.constants import UNDO_STEPS


class ImageSession:
    """
    Encapsulates a single image session, handling loading from file or clipboard,
    initial mask extraction from transparency, and maintaining all per-image state.
    """
    def __init__(self, source):
        self.source = source
        self.filename = "Clipboard" if source == "Clipboard" else os.path.basename(source)
        self.source_image = None
        self.active_image = None
        self.inherited_alpha = None
        self.source_image_np = None
        self.image_exif = None
        self.size = (0, 0)
        
        # Comprehensive state variables
        self.composite_mask = None
        self.model_output_mask = None
        self.model_output_refined = None
        self.undo_history = []
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        # Caching and refinement state
        self.composite_mask_hash = None
        self.cached_fg_corrected = None
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
        self.inherited_alpha = self._extract_source_alpha(img)
        
        # Convert to RGBA for internal processing
        self.source_image = img.convert("RGBA")
        self.active_image = self.source_image.copy()
        self.size = self.source_image.size
        
        # Create NumPy buffer for the adjustment pipeline (BGRA for OpenCV compatibility)
        self.source_image_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.source_image), cv2.COLOR_RGBA2BGRA)
        )
        
        # Initialize remaining buffers
        self.init_session_buffers(self.inherited_alpha)
        
        return self

    def init_session_buffers(self, inherited_alpha=None):
        """Initializes or resets masks, history, and scratchpads for this session."""
        if inherited_alpha:
            self.composite_mask = inherited_alpha.convert("L").copy()
        else:
            self.composite_mask = Image.new("L", self.size, 0)
            
        self.model_output_mask = None
        self.model_output_refined = None
        self.undo_history = [self._get_current_state()]
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        self.composite_mask_hash = None
        self.cached_fg_corrected = None
        self.cached_blurred_bg = None
        self.last_blur_params = None
        self.last_trimap = None
        self.user_trimap = None
            
        # Scratchpad for painting
        self.paint_image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)

    def _extract_source_alpha(self, img):
        """Extracts the alpha channel as an initial mask if it contains transparency."""
        has_alpha = 'A' in img.mode or (img.mode == 'P' and 'transparency' in img.info)
        if has_alpha:
            alpha = img.getchannel('A')
            extrema = alpha.getextrema()
            # If min is 255, the alpha is solid white (no transparency)
            if extrema and extrema[0] < 255:
                return alpha
        return None

    def _get_current_state(self):
        """Return a deep copy of the full mask state."""
        return (
            self.composite_mask.copy() if self.composite_mask is not None else None,
            self.model_output_mask.copy() if self.model_output_mask is not None else None,
            self.model_output_refined.copy() if self.model_output_refined is not None else None,
            list(self.sam_coordinates),
            list(self.sam_labels)
        )


    def _restore_state(self, state):
        """Restore a previously saved mask state."""
        (
            self.composite_mask,
            self.model_output_mask,
            self.model_output_refined,
            self.sam_coordinates,
            self.sam_labels,
        ) = state


    def add_undo_step(self):
        """Caches the current mask state onto the undo history stack."""
        state = self._get_current_state()
        self.undo_history.append(state)

        if len(self.undo_history) > UNDO_STEPS:
            self.undo_history.pop(0)

        self.redo_history.clear()


    def undo(self):
        """Reverts the mask to the previous state in history."""
        if not self.undo_history:
            return False

        self.redo_history.append(self._get_current_state())

        previous_state = self.undo_history.pop()
        self._restore_state(previous_state)

        return True


    def redo(self):
        """Restores the next mask state from the redo history stack."""
        if not self.redo_history:
            return False

        self.undo_history.append(self._get_current_state())

        if len(self.undo_history) > UNDO_STEPS:
            self.undo_history.pop(0)

        next_state = self.redo_history.pop()
        self._restore_state(next_state)

        return True


    def is_mask_modified(self):
        """Checks if the working mask has data or history."""
        if len(self.undo_history) > 1:
            return True
        if self.composite_mask:
            # Check for non-zero pixels
            extrema = self.composite_mask.getextrema()
            if extrema and extrema[1] > 0:
                return True
        return False

    def reset_composite_mask(self):
        """Wipes the current mask and history."""
        self.add_undo_step()
        self.composite_mask = Image.new("L", self.size, 0)
        self.model_output_mask = None
        self.model_output_refined = None
        self.last_trimap = None
        self.user_trimap = None
        self.sam_coordinates = []
        self.sam_labels = []

    def copy_input_to_output(self):
        """Copies the alpha channel of the original image into the working mask."""
        if self.source_image:
            self.add_undo_step()
            self.composite_mask = self.source_image.split()[3]
            return True
        return False

    def rotate(self, pil_transpose, cv2_rotate_code):
        """Rotates the entire session's per-image state."""
        self.source_image = self.source_image.transpose(pil_transpose)
        self.active_image = self.active_image.transpose(pil_transpose)
        
        if self.source_image_np is not None:
            self.source_image_np = cv2.rotate(self.source_image_np, cv2_rotate_code)
            
        if self.composite_mask:
            self.composite_mask = self.composite_mask.transpose(pil_transpose)
        if self.model_output_mask:
            self.model_output_mask = self.model_output_mask.transpose(pil_transpose)
        if self.user_trimap:
            self.user_trimap = self.user_trimap.transpose(pil_transpose)
            self.last_trimap = np.array(self.user_trimap)
            
        self.size = self.source_image.size
        # Clear state that is no longer valid for the new orientation
        # undo could probably be implemented
        self.undo_history = [self._get_current_state()]
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        # Re-init paint scratchpad
        self.paint_image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)

    def crop(self, box):
        """Crops the session state to the given bounding box."""
        self.source_image = self.source_image.crop(box)
        self.active_image = self.active_image.crop(box)
        
        # Update the NumPy buffer from the cropped raw image, so no adjustments are baked in
        self.source_image_np = np.ascontiguousarray(
            cv2.cvtColor(np.array(self.source_image), cv2.COLOR_RGBA2BGRA)
        )
        
        if self.composite_mask:
            self.composite_mask = self.composite_mask.crop(box)
        if self.model_output_mask:
            self.model_output_mask = self.model_output_mask.crop(box)
        if self.user_trimap:
            self.user_trimap = self.user_trimap.crop(box)
            self.last_trimap = np.array(self.user_trimap)
            
        self.size = self.active_image.size

        # Since dimensions changed, old undo history is invalid
        # Possibly could be implemented properly
        self.undo_history = [self._get_current_state()]
        self.redo_history = []
        self.sam_coordinates = []
        self.sam_labels = []
        
        # Re-initialise the paint scratchpad
        self.paint_image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32_Premultiplied)
        self.paint_image.fill(Qt.GlobalColor.transparent)


