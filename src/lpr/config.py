import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models", "License_plate")

# Model paths
VEHICLE_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n.pt")
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "plate.pt")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")

# Confidence thresholds
VEHICLE_CONF_THRESHOLD = 0.5
PLATE_CONF_THRESHOLD = 0.5
OCR_CONF_THRESHOLD = 0.4

# Inference Configuration
DEVICE = "cpu"  # Change to "0" or "cuda" if using GPU
