"""
Fire Detection — Evaluation Script
====================================
Evaluates a trained fire detection model on a test dataset.

Usage:
    python scripts/evaluate.py --weights models/best_accuracy_model.pth
    python scripts/evaluate.py --weights models/best_accuracy_model.pth --data-dir data/fire_dataset
    python scripts/evaluate.py --weights models/best_accuracy_model.pth --save-report models/confusion_matrix.png
"""

import sys
import os
import argparse
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.fire_detection.config import config
from src.fire_detection.model import MobileNetFireDetection
from src.fire_detection.dataset import Fire_Dataset
from src.fire_detection.dataloader import get_dataloaders
from src.fire_detection.preprocessing import get_mean_std, augmentation_main_transform
from src.fire_detection.evaluate import (
    evaluate_model,
    compute_metrics,
    print_classification_report,
    plot_confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Fire Detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py --weights models/best_accuracy_model.pth
  python scripts/evaluate.py --weights models/best_accuracy_model.pth --threshold 0.6
  python scripts/evaluate.py --weights models/best_accuracy_model.pth --save-report models/confusion_matrix.png
        """
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pth file)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'Path to dataset root (default: {config.DATA_DIR})')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation (default: 32)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--save-report', type=str, default=None,
                        help='Path to save confusion matrix image (optional)')
    
    return parser.parse_args()


def load_model(weights_path, device):
    """Load trained model from checkpoint."""
    model = MobileNetFireDetection(pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'val_accuracy' in checkpoint:
            print(f"  Checkpoint val accuracy: {checkpoint['val_accuracy']:.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device).eval()
    return model


def main():
    args = parse_args()
    
    # Validate weights file exists
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)
    
    data_dir = args.data_dir or config.DATA_DIR
    device = config.DEVICE
    
    print("=" * 60)
    print("FIRE DETECTION — EVALUATION")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Weights:    {args.weights}")
    print(f"  Data Dir:   {data_dir}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Batch Size: {args.batch_size}")
    print("=" * 60)
    
    # 1. Load dataset (no transforms — applied by Subset_Dataset)
    print("\nLoading dataset...")
    dataset = Fire_Dataset(
        root_dir=data_dir,
        transform=None,
        frames_per_video=config.FRAMES_PER_VIDEO
    )
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset) == 0:
        print("Error: No samples found in dataset!")
        sys.exit(1)
    
    # 2. Compute normalization stats
    print("Computing normalization statistics...")
    mean, std = get_mean_std(dataset)
    
    # 3. Build transforms (only need val/test transform for evaluation)
    main_transform, fire_train_transform, no_fire_train_transform = augmentation_main_transform(mean, std)
    
    # 4. Create data loaders — we use the test split for evaluation
    print("Creating data loaders...")
    _, _, test_loader = get_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        train_fac=config.TRAIN_SPLIT,
        val_fac=config.VAL_SPLIT,
        test_fac=config.TEST_SPLIT,
        main_transform=main_transform,
        fire_train_transform=fire_train_transform,
        no_fire_train_transform=no_fire_train_transform
    )
    print(f"  Test batches: {len(test_loader)}")
    
    # 5. Load model
    print(f"\nLoading model from {args.weights}...")
    model = load_model(args.weights, device)
    print("  Model loaded successfully")
    
    # 6. Run evaluation
    print("\nRunning evaluation...")
    labels, predictions, probabilities = evaluate_model(
        model, test_loader, device, threshold=args.threshold
    )
    
    # 7. Compute and display metrics
    metrics = compute_metrics(labels, predictions)
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    
    # 8. Detailed classification report
    print_classification_report(labels, predictions)
    
    # 9. Confusion matrix
    save_path = args.save_report
    if save_path is None:
        save_path = os.path.join(config.MODELS_DIR, "confusion_matrix.png")
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    cm = plot_confusion_matrix(labels, predictions, save_path=save_path)
    
    if cm is not None:
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")
    
    print("\n" + "=" * 50)
    print("Evaluation complete.")
    print("=" * 50)


if __name__ == "__main__":
    main()
