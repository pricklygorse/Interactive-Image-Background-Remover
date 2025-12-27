# Paint brush is fixed screen size for simplicity, user zooms into image to use the paint brush
# as a 'relative' smaller brush
PAINT_BRUSH_SCREEN_SIZE = 50 

# for mask edge refinement
SOFTEN_RADIUS = 2 

DEFAULT_ZOOM_FACTOR = 1.15

UNDO_STEPS = 20

# Segment anything constants
MIN_SAM_BOX_SIZE = 10
SAM_TRT_WARMUP_POINTS = 30  # how many interactive points to pre-compile for TensorRT