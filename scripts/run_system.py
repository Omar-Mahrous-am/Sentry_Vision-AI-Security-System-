"""
Fire Detection — Run System (Inference)
========================================
Runs fire detection inference on images, videos, or live webcam.

Usage:
    python scripts/run_system.py --mode image --source test.jpg --weights models/best_accuracy_model.pth
    python scripts/run_system.py --mode video --source fire_clip.mp4 --weights models/best_accuracy_model.pth
    python scripts/run_system.py --mode webcam --weights models/best_accuracy_model.pth
"""

import sys
import os
import argparse
import time

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.fire_detection.config import config
from src.fire_detection.model import MobileNetFireDetection


# ─── Shared Utilities ────────────────────────────────────────────

def get_preprocess():
    """Standard preprocessing pipeline matching training."""
    return transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
    ])


def load_model(weights_path, device):
    """Load trained model from checkpoint."""
    model = MobileNetFireDetection(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device).eval()
    return model


def predict_frame(model, frame_rgb, preprocess, device, threshold=0.5):
    """
    Run prediction on a single RGB frame (numpy array or PIL Image).
    
    Returns:
        label: 'FIRE' or 'NO FIRE'
        probability: float confidence score
    """
    if isinstance(frame_rgb, np.ndarray):
        img = Image.fromarray(frame_rgb)
    else:
        img = frame_rgb
    
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probability = torch.sigmoid(output).item()
    
    label = "FIRE" if probability >= threshold else "NO FIRE"
    return label, probability


def draw_overlay(frame, label, probability):
    """Draw prediction overlay on a BGR OpenCV frame."""
    h, w = frame.shape[:2]
    
    # Colors: red for fire, green for no fire
    if label == "FIRE":
        color = (0, 0, 255)       # Red in BGR
        bar_color = (0, 0, 200)
    else:
        color = (0, 200, 0)       # Green in BGR
        bar_color = (0, 160, 0)
    
    # Semi-transparent background bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Label text
    text = f"{label}  ({probability:.1%})"
    cv2.putText(frame, text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, color, 3, cv2.LINE_AA)
    
    # Confidence bar
    bar_width = int((w - 30) * probability)
    cv2.rectangle(frame, (15, 55), (15 + bar_width, 65), bar_color, -1)
    cv2.rectangle(frame, (15, 55), (w - 15, 65), (100, 100, 100), 1)
    
    # Fire border warning
    if label == "FIRE":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    
    return frame


# ─── Mode: Image ─────────────────────────────────────────────────

def run_image(source, model, preprocess, device, threshold, output_path=None):
    """Run fire detection on a single image."""
    if not os.path.exists(source):
        print(f"Error: Image not found: {source}")
        return
    
    # Load and predict
    img = Image.open(source).convert('RGB')
    label, probability = predict_frame(model, img, preprocess, device, threshold)
    
    print(f"\n  Result:     {label}")
    print(f"  Confidence: {probability:.4f} ({probability:.1%})")
    
    # Annotate and display/save
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    frame = draw_overlay(frame, label, probability)
    
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"  Saved to:   {output_path}")
    else:
        cv2.imshow("Fire Detection", frame)
        print("  Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─── Mode: Video ─────────────────────────────────────────────────

def run_video(source, model, preprocess, device, threshold, output_path=None):
    """Run fire detection on a video file."""
    if not os.path.exists(source):
        print(f"Error: Video not found: {source}")
        return
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {source}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps}")
    print(f"  Frames:     {total_frames}")
    
    # Setup video writer if saving
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    fire_count = 0
    
    print("\n  Processing... (press 'q' to stop)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert BGR to RGB for prediction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label, probability = predict_frame(model, frame_rgb, preprocess, device, threshold)
        
        if label == "FIRE":
            fire_count += 1
        
        # Draw overlay on original BGR frame
        frame = draw_overlay(frame, label, probability)
        
        # Add frame counter
        cv2.putText(frame, f"Frame {frame_count}/{total_frames}", 
                    (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        if writer:
            writer.write(frame)
        
        cv2.imshow("Fire Detection — Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("  Stopped by user.")
            break
    
    cap.release()
    if writer:
        writer.release()
        print(f"  Output saved to: {output_path}")
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n  Processed:   {frame_count} frames")
    print(f"  Fire frames: {fire_count} ({fire_count/max(frame_count,1)*100:.1f}%)")


# ─── Mode: Webcam ────────────────────────────────────────────────

def run_webcam(model, preprocess, device, threshold, output_path=None):
    """Run live fire detection on webcam feed."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer if saving
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, 20, (width, height))
    
    print(f"  Webcam resolution: {width}x{height}")
    print("  Press 'q' to quit.\n")
    
    frame_count = 0
    fire_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Predict
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label, probability = predict_frame(model, frame_rgb, preprocess, device, threshold)
        
        if label == "FIRE":
            fire_count += 1
        
        # Draw overlay
        frame = draw_overlay(frame, label, probability)
        
        # FPS counter
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        if writer:
            writer.write(frame)
        
        cv2.imshow("Fire Detection — Live", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if writer:
        writer.release()
        print(f"  Recording saved to: {output_path}")
    cv2.destroyAllWindows()
    
    elapsed = time.time() - start_time
    print(f"\n  Session: {elapsed:.1f}s, {frame_count} frames, avg {frame_count/max(elapsed,1):.1f} FPS")
    print(f"  Fire detections: {fire_count}/{frame_count} frames")


# ─── CLI Entry Point ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fire Detection — Inference System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_system.py --mode image --source photo.jpg --weights models/best_accuracy_model.pth
  python scripts/run_system.py --mode video --source clip.mp4 --weights models/best_accuracy_model.pth --output result.mp4
  python scripts/run_system.py --mode webcam --weights models/best_accuracy_model.pth
        """
    )
    
    parser.add_argument('--mode', type=str, required=True, choices=['image', 'video', 'webcam'],
                        help='Inference mode: image, video, or webcam')
    parser.add_argument('--source', type=str, default=None,
                        help='Path to input image or video file (required for image/video modes)')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save annotated output (optional)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate args
    if args.mode in ('image', 'video') and args.source is None:
        print(f"Error: --source is required for '{args.mode}' mode.")
        sys.exit(1)
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    device = config.DEVICE
    
    print("=" * 60)
    print("FIRE DETECTION — INFERENCE SYSTEM")
    print("=" * 60)
    print(f"  Device:    {device}")
    print(f"  Mode:      {args.mode}")
    print(f"  Weights:   {args.weights}")
    print(f"  Threshold: {args.threshold}")
    if args.source:
        print(f"  Source:    {args.source}")
    if args.output:
        print(f"  Output:    {args.output}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.weights, device)
    preprocess = get_preprocess()
    print("  Model loaded successfully.\n")
    
    # Run selected mode
    if args.mode == 'image':
        run_image(args.source, model, preprocess, device, args.threshold, args.output)
    elif args.mode == 'video':
        run_video(args.source, model, preprocess, device, args.threshold, args.output)
    elif args.mode == 'webcam':
        run_webcam(model, preprocess, device, args.threshold, args.output)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
