import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models", "weapon_detection")

# Model paths
WEAPON_MODEL_PATH = os.path.join(MODELS_DIR, "best.pt")  