from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QCheckBox, QFileDialog, 
                             QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QSlider, QFrame, QSplitter, QDialog, QScrollArea, 
                             QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem, 
                             QTextEdit, QSizePolicy, QRadioButton, QButtonGroup, QInputDialog, 
                             QProgressBar, QStyle,
                             QListWidget, QListWidgetItem, QListView, QAbstractItemView)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QSettings, QPropertyAnimation, QEasingCurve, QSize
from PyQt6.QtGui import (QPixmap, QImage, QColor, QPainter, QPainterPath, QPen, QBrush,
                         QKeySequence, QShortcut, QCursor, QIcon, QPalette)

import os


from src.constants import DEFAULT_ZOOM_FACTOR, PAINT_BRUSH_SCREEN_SIZE, MIN_SAM_BOX_SIZE




# Animated Collapsable Frame Widget
class CollapsibleFrame(QFrame):
    def __init__(self, title="Options", parent=None, animation_duration=250, tooltip=None):
        super().__init__(parent)
        self.animation_duration = animation_duration
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0) # No margin on the outer frame
        self.main_layout.setSpacing(0)
        
        # 1. Header Button
        self.toggle_button = QPushButton(title)
        if tooltip:
            self.toggle_button.setToolTip(tooltip)
        # setting this style overrides dark mode palette
        #self.toggle_button.setStyleSheet("text-align: left; padding: 5px;")
        self.toggle_button.setFlat(True)
        self.toggle_button.clicked.connect(self.toggle_content)
        
        # 2. Content Frame
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        self.content_layout.setContentsMargins(6, 6, 6, 6)
        self.content_layout.setSpacing(5)
        
        # 3. Animation Setup
        self.animation = QPropertyAnimation(self.content_frame, b"maximumHeight")
        self.animation.setDuration(self.animation_duration)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        # 4. Add to Main Layout
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_frame)
        
        # Set initial state to collapsed (but hide/show the contents)
        self.is_collapsed = True
        self.content_frame.setVisible(False)
        self.content_frame.setMaximumHeight(0)
        
        self.toggle_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarUnshadeButton))

    def layout_for_content(self):
        """Returns the inner layout where the user's widgets should be added."""
        return self.content_layout
            
    def toggle_content(self, checked=None):
        self.animation.stop()

        if self.is_collapsed:
            self.is_collapsed = False
            self.content_frame.setVisible(True) 
            self.toggle_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarShadeButton))
            
            collapsed_height = self.content_frame.height()
            self.content_frame.setMaximumHeight(collapsed_height)
            
            # Set the end value to a very large number, 
            # and the layout manager will clamp it to the required size.
            target_height = 1000 
            
            self.animation.setStartValue(collapsed_height)
            self.animation.setEndValue(target_height)
            self.animation.start()
        else:
            self.is_collapsed = True
            self.toggle_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarUnshadeButton))
            
            start_height = self.content_frame.height()
            
            self.animation.setStartValue(start_height)
            self.animation.setEndValue(0) # Collapse to 0
            
            self.animation.finished.connect(lambda: self.content_frame.setVisible(False), 
                                           Qt.ConnectionType.SingleShotConnection)
            
            self.animation.start()

        self.collapsible_set_light_dark()


    def collapsible_set_light_dark(self):
        """Updates the expand/collapse icon, inverting it for dark mode."""

        pixmap_enum = (
            QStyle.StandardPixmap.SP_TitleBarUnshadeButton
            if self.is_collapsed
            else QStyle.StandardPixmap.SP_TitleBarShadeButton
        )
        
        icon = self.style().standardIcon(pixmap_enum)

        is_dark_mode = QApplication.instance().palette().color(QPalette.ColorRole.Window).lightness() < 128

        if is_dark_mode:
            pixmap = icon.pixmap(self.toggle_button.iconSize())
            if not pixmap.isNull():
                img = pixmap.toImage()
                img.invertPixels(QImage.InvertMode.InvertRgb)
                icon = QIcon(QPixmap.fromImage(img))
            # lighten the border
            self.setStyleSheet("""
                CollapsibleFrame {
                    border: 1px solid #707070; /* Visible gray border */
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("")


        self.toggle_button.setIcon(icon)



# Controls mouse clicks, panning, scrolling etc on the input and output views

class SynchronisedGraphicsView(QGraphicsView):
    def __init__(self, scene, name="View", parent=None):
        super().__init__(scene, parent)
        self.name = name
        
        self.setMouseTracking(True) 
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.sibling = None
        self.panning = False
        self.pan_start = QPointF()
        self.controller = None
        
        self.is_painting = False
        self.last_paint_pos = None
        self.box_start = None
        self.temp_box_item = None
        
        self.brush_cursor_item = QGraphicsEllipseItem(0, 0, 10, 10)
        self.brush_cursor_item.setPen(QPen(QColor(0, 0, 0, 255), 1)) 
        self.brush_cursor_outer = QGraphicsEllipseItem(0, 0, 10, 10, self.brush_cursor_item)
        self.brush_cursor_outer.setPen(QPen(QColor(255, 255, 255, 255), 1))
        
        self.brush_cursor_item.setBrush(QBrush(Qt.BrushStyle.NoBrush)) 
        
        self.brush_cursor_item.setZValue(99999) 
        
        self.scene().addItem(self.brush_cursor_item)
        self.brush_cursor_item.hide()

    def drawBackground(self, painter, rect):
        """
        Overrides the background drawing to keep the hatch pattern 
        fixed to screen pixels, preventing distortion during zoom/pan.
        """
        painter.save()

        painter.resetTransform()

        viewport_rect = self.viewport().rect()

        painter.fillRect(viewport_rect, self.palette().color(QPalette.ColorRole.Base))
        painter.fillRect(viewport_rect, self.backgroundBrush())
        
        painter.restore()

    def dragEnterEvent(self, event):
        # Pass the event to the parent to handle
        event.ignore()

    def dropEvent(self, event):
        # Pass the event to the parent to handle
        event.ignore()

    def set_sibling(self, sibling_view):
        self.sibling = sibling_view
        self.horizontalScrollBar().valueChanged.connect(self.sync_scroll_h)
        self.verticalScrollBar().valueChanged.connect(self.sync_scroll_v)

    def set_controller(self, ctrl):
        self.controller = ctrl

    def sync_scroll_h(self, value):
        if self.sibling and self.sibling.horizontalScrollBar().value() != value:
            self.sibling.horizontalScrollBar().setValue(value)

    def sync_scroll_v(self, value):
        if self.sibling and self.sibling.verticalScrollBar().value() != value:
            self.sibling.verticalScrollBar().setValue(value)

    def update_point_scales(self):
        # Get the X scaling factor (m11) from the transform matrix
        zoom = self.transform().m11()
        
        if zoom == 0: zoom = 1
        
        # If zoom is 2.0 (200%), scale should be 0.5 to look normal
        inverse_scale = 1.0 / zoom
        
        for item in self.scene().items():
            # Check for the tag we set in handle_sam_point
            if item.data(0) == "sam_point":
                item.setScale(inverse_scale)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        is_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        
        is_touchpad_gesture = (event.phase() != Qt.ScrollPhase.NoScrollPhase) or not pixel_delta.isNull()
        
        if is_touchpad_gesture and not is_ctrl:
            # --- PANNING ---
            
            # Calculate movement vector
            if not pixel_delta.isNull():
                # Use exact pixels from trackpad
                dx = pixel_delta.x()
                dy = pixel_delta.y()
            else:
                # Fallback: Convert angle notches to pixels
                # Standard mouse wheel step is 120. We scale it down for panning.
                dx = angle_delta.x() / 8
                dy = angle_delta.y() / 8

            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            
            # Subtract delta to pan naturally
            hs.setValue(int(hs.value() - dx))
            vs.setValue(int(vs.value() - dy))
            
            # Sync Sibling
            if self.sibling:
                self.sibling.horizontalScrollBar().setValue(hs.value())
                self.sibling.verticalScrollBar().setValue(vs.value())
            
            event.accept()
            return

        # --- ZOOMING ---
        
        target_viewport_pos = event.position().toPoint()
        target_scene_pos = self.mapToScene(target_viewport_pos)
        
        delta = angle_delta.y()
        if delta == 0: return
        
        zoom_in = delta > 0
        factor = DEFAULT_ZOOM_FACTOR if zoom_in else 1 / DEFAULT_ZOOM_FACTOR
        
        current_scale = self.transform().m11()
        
        view_rect = self.viewport().rect()
        scene_rect = self.sceneRect()
        
        if scene_rect.width() > 0 and scene_rect.height() > 0:
            ratio_w = view_rect.width() / scene_rect.width()
            ratio_h = view_rect.height() / scene_rect.height()
            min_scale = min(ratio_w, ratio_h)
        else:
            min_scale = 0.01 
            
        max_scale = 10.0 
        
        new_scale = current_scale * factor
        
        if new_scale > max_scale:
            factor = max_scale / current_scale
            if factor <= 1.0 and zoom_in: return 
        elif new_scale < min_scale:
            factor = min_scale / current_scale
            if factor >= 1.0 and not zoom_in: return 
        
        self.scale(factor, factor)
        self.update_point_scales()
        
        # Recenter zoom on mouse
        new_viewport_pos = self.mapFromScene(target_scene_pos)
        delta_viewport = new_viewport_pos - target_viewport_pos
        
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta_viewport.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta_viewport.y())
        
        if self.sibling:
            self.sibling.setTransform(self.transform())
            self.sibling.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
            self.sibling.verticalScrollBar().setValue(self.verticalScrollBar().value())
            self.sibling.update_point_scales()
            
        if self.controller:
            self.controller.update_zoom_label()
            self.update_brush_cursor(target_scene_pos)
            if self.sibling:
                self.sibling.update_brush_cursor(target_scene_pos)
        
        event.accept()

    def keyPressEvent(self, event):
        pan_step = 20 # Pixels to move
        if event.key() == Qt.Key.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - pan_step)
        elif event.key() == Qt.Key.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + pan_step)
        elif event.key() == Qt.Key.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - pan_step)
        elif event.key() == Qt.Key.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + pan_step)
        else:
            super().keyPressEvent(event)

    def update_brush_cursor(self, scene_pos):
        if not self.controller or not self.controller.paint_mode:
            if self.brush_cursor_item.isVisible():
                self.brush_cursor_item.hide()
            return
        
        if not self.brush_cursor_item.isVisible():
            self.brush_cursor_item.show()
            
        zoom = self.transform().m11()
        if zoom == 0: zoom = 1
        scene_dia = PAINT_BRUSH_SCREEN_SIZE / zoom
        r = scene_dia / 2
        
        rect_centered = QRectF(-r, -r, scene_dia, scene_dia)
        self.brush_cursor_item.setRect(rect_centered)
        self.brush_cursor_outer.setRect(rect_centered)
        self.brush_cursor_item.setPos(scene_pos)
        
        pw = max(1.0, 1.0 / zoom)
        self.brush_cursor_item.setPen(QPen(QColor(0,0,0), pw))
        self.brush_cursor_outer.setPen(QPen(QColor(255,255,255), pw, Qt.PenStyle.DashLine))

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self.controller:
            if self.controller.paint_mode:
                if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
                    self.is_painting = True
                    self.last_paint_pos = scene_pos
                    self.controller.handle_paint_start(scene_pos)
                    event.accept()
                    return
            else:
                if event.button() == Qt.MouseButton.LeftButton:
                    self.box_start = scene_pos
                    self.temp_box_item = QGraphicsRectItem()
                    zoom = self.transform().m11()
                    pen_width = 2.0 / zoom if zoom != 0 else 2.0
                    self.temp_box_item.setPen(QPen(Qt.GlobalColor.red, pen_width))
                    self.scene().addItem(self.temp_box_item)
                    self.temp_box_item.hide()
                    event.accept()
                elif event.button() == Qt.MouseButton.RightButton:
                    self.controller.handle_sam_point(scene_pos, is_positive=False)
                    event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        
        self.update_brush_cursor(scene_pos)
        if self.sibling:
            self.sibling.update_brush_cursor(scene_pos)

        if self.panning:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            hs.setValue(hs.value() - int(delta.x()))
            vs.setValue(vs.value() - int(delta.y()))
            event.accept()
            return
            
        if self.controller and self.controller.paint_mode and self.is_painting:
            self.controller.handle_paint_move(self.last_paint_pos, scene_pos)
            self.last_paint_pos = scene_pos
            event.accept()
            return
            
        if self.box_start and self.temp_box_item:
            rect = QRectF(self.box_start, scene_pos).normalized()
            self.temp_box_item.setRect(rect)
            zoom = self.transform().m11()
            if zoom == 0: zoom = 1
            min_size_in_scene = MIN_SAM_BOX_SIZE / zoom

            if rect.width() >= min_size_in_scene or rect.height() >= min_size_in_scene:
                self.temp_box_item.show()
                for item in self.scene().items():
                    if item.data(0) == "sam_point":
                        self.scene().removeItem(item)
            else:
                self.temp_box_item.hide()
            event.accept()
            return
            
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        # Handle Panning Release
        if self.panning and event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
        # Handle Paint Release (Left OR Right button)
        if self.controller and self.controller.paint_mode:
            if event.button() == Qt.MouseButton.LeftButton or event.button() == Qt.MouseButton.RightButton:
                self.is_painting = False
                self.last_paint_pos = None
                self.controller.handle_paint_end()
                
        # Handle Box Selection Release (Left Only)
        if self.box_start and event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            rect = QRectF(self.box_start, scene_pos).normalized()
            if self.temp_box_item:
                self.scene().removeItem(self.temp_box_item)
                self.temp_box_item = None
            self.box_start = None
            
            # Convert MIN_RECT_SIZE from screen pixels to scene pixels
            zoom = self.transform().m11()
            if zoom == 0: zoom = 1
            min_size_in_scene = MIN_SAM_BOX_SIZE / zoom

            # Logic to distinguish between a Click (Point) and a Drag (Box)
            if rect.width() < min_size_in_scene and rect.height() < min_size_in_scene:
                if self.controller: 
                    # If we are in paint mode, clicks are paint dots, not SAM points
                    if not self.controller.paint_mode:
                        self.controller.handle_sam_point(scene_pos, is_positive=True)
            else:
                if self.controller and not self.controller.paint_mode: 
                    self.controller.handle_sam_box(rect)
                    
        super().mouseReleaseEvent(event)
        
    def leaveEvent(self, event):
        # --- Clean hide on both screens when leaving ---
        if self.controller and self.controller.paint_mode:
            self.brush_cursor_item.hide()
            if self.sibling:
                # Direct access to sibling item to ensure sync
                self.sibling.brush_cursor_item.hide()
        super().leaveEvent(event)




class ThumbnailList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setWrapping(False)
        self.setFixedHeight(120)
        self.setSpacing(2)
        self.setIconSize(QSize(70, 70))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.update_style()

    def update_style(self, is_dark=False):
        text_color = "white" if is_dark else "black"
        self.setStyleSheet(f"""
            QListWidget {{
                border: none;
                background-color: transparent;
                color: {text_color};
            }}
            QListWidget::item {{
                border: 2px solid transparent;
                border-radius: 5px;
                color: {text_color};
            }}
            QListWidget::item:selected {{
                border: 2px solid #2a82da;
                background-color: rgba(42, 130, 218, 0.2);
            }}
        """)

    def wheelEvent(self, event):
        """Redirects vertical mouse wheel movement to horizontal scrolling."""
        # Check if the event has a vertical delta (typical mouse wheel)
        delta = event.angleDelta().y()
        if delta != 0:
            # Scroll the horizontal scroll bar instead
            # Subtracting the delta makes 'wheel down' scroll right
            hs = self.horizontalScrollBar()
            hs.setValue(hs.value() - delta)
            event.accept()
        else:
            super().wheelEvent(event)

    def add_image_thumbnail(self, file_path):
        if file_path == "Clipboard":
            # Just a placeholder icon for clipboard
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            item = QListWidgetItem(icon, "Clipboard")
        else:
            # Generate a fast thumbnail using QPixmap
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                thumb = pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                b_name = os.path.basename(file_path)
                if len(b_name) > 16:
                    b_name = f"{b_name[:7]}...{b_name[-7:]}"

                item = QListWidgetItem(QIcon(thumb), b_name)
            else:
                return
        
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        item.setToolTip(file_path)
        self.addItem(item)