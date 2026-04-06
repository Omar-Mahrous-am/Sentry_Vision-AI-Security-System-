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

# --- 2. The Inference Engine ---
def run_inference(image_path, model_path, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model — uses the SAME architecture as training
    model = MobileNetFireDetection(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle if you saved the whole model or just the state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device).eval()
    
    # Load and Transform Image
    img = Image.open(image_path).convert('RGB')
    preprocess = get_preprocess()
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
    
    label = "FIRE" if probability >= threshold else "NO FIRE"
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
        result, confidence = run_inference(args.image, args.weights, args.threshold)
        print(f"\nResult: **{result}**")
        print(f"Confidence: {confidence:.4f}")
        print("-" * 30)