# TODO: Refactor to YAML

# LM Value mapping
VALUE_MAP = {1: 40, 2: 80, 3: 120, 4: 200}


# Visualization and debugging
SHOW_MINED = True
SHOW_VALID = True

# Overlap option
READ_TYPE = 'sequential'  # Change to 'sequential' for increased effiency, 'random' for random calls
OVERLAP_FACTOR = 0.0  # Portion of patches that are allowed to overlap (0->1)

# Misc
WHITE_COLOR = 250
SCALE = 16
PATCH_SIZE = (256, 256)
NUM_WORKERS = 100
NUM_PATCHES = -1 # -1 to mine until exhaustion


# RGB Masking
PEN_SIZE_THRESHOLD = 200
MINIMUM_COLOR_DIFFERENCE = 30
BGR_RED_CHANNEL = 2
BGR_GREEN_CHANNEL = 1
BGR_BLUE_CHANNEL = 0
PEN_MASK_EXPANSION = 9

# HSV Masking
HSV_MASK_S_THRESHOLD = 15
HSV_MASK_V_THRESHOLD = 90

