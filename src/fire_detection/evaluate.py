import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    Run model on a DataLoader and collect all predictions + ground truth labels.
    
    Returns:
        all_labels: list of ground truth labels (0 or 1)
        all_preds: list of predicted labels (0 or 1)
        all_probs: list of predicted probabilities
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.float()
            
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs).cpu().squeeze()
            predictions = (probabilities >= threshold).float()
            
            # Handle single-item batches
            if probabilities.dim() == 0:
                probabilities = probabilities.unsqueeze(0)
                predictions = predictions.unsqueeze(0)
            
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(predictions.numpy().tolist())
            all_probs.extend(probabilities.numpy().tolist())
    
    return all_labels, all_preds, all_probs


def compute_metrics(labels, predictions):
    """
    Compute classification metrics.
    
    Returns:
        dict with accuracy, precision, recall, f1
    """
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1_score': f1_score(labels, predictions, zero_division=0),
    }


def print_classification_report(labels, predictions):
    """Print a detailed classification report."""
    target_names = ['No Fire', 'Fire']
    report = classification_report(labels, predictions, target_names=target_names, zero_division=0)
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(report)
    return report


def plot_confusion_matrix(labels, predictions, save_path=None):
    """
    Generate and optionally save a confusion matrix plot.
    
    Args:
        labels: ground truth labels
        predictions: predicted labels
        save_path: if provided, saves the plot to this path
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Skipping confusion matrix plot.")
        return None
    
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Fire', 'Fire'],
                yticklabels=['No Fire', 'Fire'],
                ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix — Fire Detection', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close(fig)
    return cm
