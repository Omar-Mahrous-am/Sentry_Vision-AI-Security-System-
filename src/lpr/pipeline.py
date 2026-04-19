import cv2
import numpy as np

from .config import VEHICLE_MODEL_PATH, PLATE_MODEL_PATH, OCR_MODEL_PATH, DEVICE, PLATE_CONF_THRESHOLD, OCR_CONF_THRESHOLD
from .detector import LPRDetector
from .mapper import is_arabic_letter, is_digit, classify_governorate

class LPRPipeline:
    def __init__(self):
        # Initialize detector components
        self.detector = LPRDetector(
            vehicle_model_path=VEHICLE_MODEL_PATH,
            plate_model_path=PLATE_MODEL_PATH,
            ocr_model_path=OCR_MODEL_PATH,
            device=DEVICE
        )

    def process_image(self, image_data):
        """
        Process an image array or bytes and return LPR results.
        image_data: image bytes or np.ndarray (OpenCV format)
        """
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        if img is None:
            raise ValueError("Invalid image data provided for LPR.")

        results = {
            "plates": []
        }

        # 1. Detect License Plates
        # Note: If vehicle detection is necessary, we'd loop over vehicles first.
        # Based on pipeline diagram, we input Frame -> Plate directly.
        plates = self.detector.detect_plates(img, conf=PLATE_CONF_THRESHOLD)

        for plate in plates:
            px1, py1, px2, py2 = plate["bbox"]
            plate_conf = plate["confidence"]

            # Crop the plate region
            # Ensure coordinates are within image boundaries
            h, w = img.shape[:2]
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(w, px2), min(h, py2)

            cropped_plate = img[py1:py2, px1:px2]
            
            if cropped_plate.size == 0:
                continue

            # 2. Detect individual Arabic characters & digits on cropped plate
            ocr_elements = self.detector.detect_ocr(cropped_plate, conf=OCR_CONF_THRESHOLD)

            # 3. Sort by position (Right-to-Left for Arabic geometry)
            # x1 is the coordinate from the left edge.
            # To sort from right to left, sort by x1 descending.
            ocr_elements.sort(key=lambda item: item["bbox"][0], reverse=True)

            letters = []
            digits = []
            ocr_text = []

            for elem in ocr_elements:
                char_name = elem["class_name"]
                ocr_text.append(char_name)

                # Segment into letters and digits for classification
                if is_digit(char_name):
                    digits.append(char_name)
                elif is_arabic_letter(char_name):
                    letters.append(char_name)
                else:
                    # In case the model returns english string class names that represent arabic letters
                    letters.append(char_name)

            final_text_str = " ".join(ocr_text)

            # 4. Governorate Classifier
            governorate = classify_governorate(letters, digits)

            results["plates"].append({
                "bbox": [px1, py1, px2, py2],
                "text": final_text_str,
                "confidence": plate_conf, # plate confidence
                "governorate": governorate,
                "details": {
                    "letters": letters,
                    "digits": digits
                }
            })

        return results

# Singleton pipeline instance for inference routing
_lpr_pipeline = None

def get_lpr_pipeline():
    global _lpr_pipeline
    if _lpr_pipeline is None:
        _lpr_pipeline = LPRPipeline()
    return _lpr_pipeline
