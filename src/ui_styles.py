from PyQt6.QtGui import QPalette, QColor, QBrush, QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QStyle

def get_theme_palette(mode):
    """Returns a QPalette for Light or Dark mode."""
    palette = QPalette()
    
    if mode == 'dark':
        dark_color = QColor(45, 45, 45)
        disabled_color = QColor(127, 127, 127)
        palette.setColor(QPalette.ColorRole.Window, dark_color)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, dark_color)
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, dark_color)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_color)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(80, 80, 80))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, disabled_color)
    else:
        light_color = QColor(240, 240, 240)            
        palette.setColor(QPalette.ColorRole.Window, light_color)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.Button, light_color)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        
        disabled_text = QColor(120, 120, 120)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, QColor(150, 150, 150))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)

    return palette


def apply_theme(main_ui_window, mode):

    app = QApplication.instance()
    if not app: return

    app.setStyle("Fusion")
    palette = get_theme_palette(mode)
    app.setPalette(palette)

    # adjust checkerboard backgrounds
    main_ui_window.view_input.update_background_theme()
    main_ui_window.view_output.update_background_theme()

    # Stylesheet overrides
    if mode == 'dark':
        commit_style = """
            QFrame { background-color: rgba(255, 255, 255, 0.05); border: 1px solid #555; border-radius: 4px; }
            QLabel { color: white; background: transparent; border: none; }
        """
    else:
        commit_style = """
            QFrame { background-color: rgba(0, 0, 0, 0.05); border: 1px solid #ccc; border-radius: 4px; }
            QLabel { color: black; background: transparent; border: none; }
        """
    
    main_ui_window.mask_action_panel.setStyleSheet(commit_style)
    
    # Update components that have internal theme logic
    if hasattr(main_ui_window, 'hw_options_frame'):
        main_ui_window.hw_options_frame.collapsible_set_light_dark()
    
    if hasattr(main_ui_window, 'thumbnail_strip'):
        main_ui_window.thumbnail_strip.update_style(mode == 'dark')
        
    # Refresh the splitter orientation (which handles its own icon/direction)
    main_ui_window.toggle_splitter_orientation(initial_setup=True)