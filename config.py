BASE_URL = 'http://127.0.0.1:5001'

# Output folder where images are saved
WRITE_PATH = './static/output/'

# Image endpoint URL
IMG_PATH = f'{BASE_URL}/static/output/'

# Features endpoint URL
FEATURES_PATH = f'{BASE_URL}/mask/features?image={BASE_URL}/static/output/'

# Minimum Confidence threshold for face landmark detection
MIN_CONF = 0.5

# Minimum Confidence threshold for face detection
MIN_CONF_FACE = 0.5

# Compression percentage for images (for downsampling)
COMPRESS_PCT = 0.3

SEGMENT_CONF = 0.8