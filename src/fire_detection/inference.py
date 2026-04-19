import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from .model import MobileNetFireDetection


# --- 1. Preprocessing Logic ---
def get_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

# --- 2. The Inference Engine class ---
class FireDetector:
    def __init__(self, model_path, device=None, threshold=0.5):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # Load Model
        self.model = MobileNetFireDetection(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device).eval()
        self.preprocess = get_preprocess()

    def detect_fire(self, image_data):
        """Processes image data (bytes or path) and returns 1 if fire is detected, else 0."""
        if isinstance(image_data, bytes):
            from io import BytesIO
            img = Image.open(BytesIO(image_data)).convert('RGB')
        elif isinstance(image_data, str):
            img = Image.open(image_data).convert('RGB')
        else:
            raise ValueError("image_data must be bytes or a file path string.")

        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probability = torch.sigmoid(output).item()
        
        return 1 if probability >= self.threshold else 0

def run_inference(image_path, model_path, threshold=0.5):
    detector = FireDetector(model_path, threshold=threshold)
    result_int = detector.detect_fire(image_path)
    
    # Maintain backward compatibility with the tuple return if needed by CLI
    # but for CLI we'll just re-calculate or adjust
    img = Image.open(image_path).convert('RGB')
    img_tensor = detector.preprocess(img).unsqueeze(0).to(detector.device)
    with torch.no_grad():
        output = detector.model(img_tensor)
        probability = torch.sigmoid(output).item()
    
    label = "FIRE" if result_int == 1 else "NO FIRE"
    return label, probability

# --- 3. CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Fire Detection Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--weights", type=str, default="fire_model.pth", help="Path to .pth file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found.")
    else:
        # Check if the weights file exists in models/fire_model/ as well
        weights = args.weights
        if not os.path.exists(weights):
            alt_weights = os.path.join("models", "fire_model", "fire_model.pth")
            if os.path.exists(alt_weights):
                weights = alt_weights
        
        result, confidence = run_inference(args.image, weights, args.threshold)
        print(f"\nResult: **{result}**")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 30)