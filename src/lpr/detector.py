import numpy as np
import cv2
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class LPRDetector:
    def __init__(self, vehicle_model_path, plate_model_path, ocr_model_path, device="cpu"):
        self.device = device
        
        try:
            # We initialize the models. Sometimes yolo11n can detect plates directly, 
            # but per architecture, we follow: yolo11n -> plate.pt -> best.pt (OCR)
            # Actually, sometimes yolo11n is optional if plate.pt is directly used,
            # but we load all specified models.
            self.vehicle_model = YOLO(vehicle_model_path) if vehicle_model_path else None
            self.plate_model = YOLO(plate_model_path)
            self.ocr_model = YOLO(ocr_model_path)
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise e

    def detect_vehicles(self, image, conf=0.5):
        """
        Detect vehicles in an image using the yolo11n model.
        Returns a list of bounding boxes [x1, y1, x2, y2].
        If no vehicle model, returns the whole image as one bounding box.
        """
        if not self.vehicle_model:
            h, w = image.shape[:2]
            return [[0, 0, w, h]]

        results = self.vehicle_model.predict(image, conf=conf, device=self.device, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                # Add class filtering if needed, e.g., only 'car', 'truck', 'bus' (usually classes 2,3,5,7 in COCO)
                # For now, we return all detected boxes assumes they are vehicles based on prompt context.
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                boxes.append(int(cls) for cls in [x1, y1, x2, y2])
        return boxes

    def detect_plates(self, image, conf=0.5):
        """
        Detect license plates in an image.
        Returns a list of dicts with bbox and confidence.
        """
        results = self.plate_model.predict(image, conf=conf, device=self.device, verbose=False)
        plates = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                plates.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence
                })
        return plates

    def detect_ocr(self, cropped_plate, conf=0.4):
        """
        Detect individual Arabic characters & digits on a cropped plate image.
        Returns a list of dicts with bbox, class_name, confidence.
        """
        results = self.ocr_model.predict(cropped_plate, conf=conf, device=self.device, verbose=False)
        characters = []
        for r in results:
            # model.names dictionary holds the class names mapping
            class_names = r.names
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Fetch string mapped name
                class_name = class_names.get(cls_id, str(cls_id))
                
                characters.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": confidence
                })
        return characters
