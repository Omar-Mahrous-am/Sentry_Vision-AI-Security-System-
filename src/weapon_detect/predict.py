import cv2
import numpy as np
from .config import WEAPON_MODEL_PATH 
from ultralytics import YOLO

class WeaponDetection:
    def __init__(self, device="cpu"):
        """
        Initialize detector components once. 
        Loading the model here ensures fast inference later.
        """
        self.device = device
        # Load the model once during initialization using ultralytics YOLO
        self.model = YOLO(WEAPON_MODEL_PATH)
        self.model.to(self.device)

    def process_weapon(self, image_data):
        """
        Convert raw input (bytes or numpy) into an OpenCV image and run detection.
        """
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        if img is None:
            raise ValueError("Invalid image data provided for weapon detection.")
            
        return self.detect_weapon(img)

    def detect_weapon(self, image):
        """
        Perform inference using the pre-loaded model.
        """
        # YOLO(image) returns a list of Results objects
        results = self.model(image, verbose=False)
        
        # Check if any bounding boxes are found in the first result
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            return results[0].boxes
        return []