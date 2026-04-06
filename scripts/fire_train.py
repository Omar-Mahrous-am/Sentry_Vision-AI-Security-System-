"""
Fire Detection — Training Script
=================================
Trains the fire detection model using MobileNetV2.

Usage:
    python scripts/fire_train.py
    python scripts/fire_train.py --epochs 30 --lr 0.0005 --batch-size 16
    python scripts/fire_train.py --data-dir data/custom_fire_data --output models/my_model.pth
"""

import sys
import os
import argparse

# Add project root to path so we can import src modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.fire_detection.config import config
from src.fire_detection.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Fire Detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fire_train.py
  python scripts/fire_train.py --epochs 30 --lr 0.0005
  python scripts/fire_train.py --data-dir data/my_data --batch-size 16
        """
    )
    
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'Path to dataset root (default: {config.DATA_DIR})')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {config.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size (default: {config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'Learning rate (default: {config.LEARNING_RATE})')
    parser.add_argument('--patience', type=int, default=None,
                        help=f'Early stopping patience (default: {config.PATIENCE})')
    parser.add_argument('--output', type=str, default=None,
                        help=f'Path to save final model (default: {config.MODEL_PATH})')
    
    return parser.parse_args()


def apply_overrides(args):
    """Apply CLI argument overrides to the global config."""
    if args.data_dir is not None:
        config.DATA_DIR = args.data_dir
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.patience is not None:
        config.PATIENCE = args.patience
    if args.output is not None:
        config.MODEL_PATH = args.output


def plot_training_history(history, save_path):
    """Save training loss/accuracy curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping training history plot.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Loss Over Epochs', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_title('Accuracy Over Epochs', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Training history plot saved to: {save_path}")


def main():
    args = parse_args()
    apply_overrides(args)
    
    # Ensure output directories exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("FIRE DETECTION — TRAINING")
    print("=" * 60)
    print(f"  Device:        {config.DEVICE}")
    print(f"  Data Dir:      {config.DATA_DIR}")
    print(f"  Epochs:        {config.EPOCHS}")
    print(f"  Batch Size:    {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Patience:      {config.PATIENCE}")
    print(f"  Model Output:  {config.MODEL_PATH}")
    print(f"  Best Model:    {config.BEST_MODEL_PATH}")
    print("=" * 60)
    
    # Run the training pipeline
    history, best_acc = run_pipeline()
    
    # Save training history plot
    history_plot_path = os.path.join(config.MODELS_DIR, "training_history.png")
    plot_training_history(history, history_plot_path)
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Validation Accuracy: {best_acc:.4f}")
    print(f"  Best Model:  {config.BEST_MODEL_PATH}")
    print(f"  Final Model: {config.MODEL_PATH}")
    print(f"  History Plot: {history_plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
