"""
Tests for src.fire_detection.train — train_model function.
Uses small synthetic data to validate training loop mechanics.
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock, patch


@pytest.fixture
def tiny_loaders():
    """
    Create two tiny DataLoaders (train + val) with synthetic data.
    4 samples each, batch_size=2, so 2 batches per loader.
    """
    x_train = torch.randn(4, 3, 224, 224)
    y_train = torch.tensor([0, 1, 1, 0])
    x_val = torch.randn(4, 3, 224, 224)
    y_val = torch.tensor([1, 0, 0, 1])
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=2)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=2)
    return train_loader, val_loader


@pytest.fixture
def training_components():
    """Return model, criterion, optimizer, scheduler for training."""
    from src.fire_detection.model import MobileNetFireDetection
    model = MobileNetFireDetection(pretrained=False)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    return model, criterion, optimizer, scheduler


class TestTrainModel:
    """Tests for the train_model function."""

    def test_returns_history_and_accuracy(self, tiny_loaders, training_components, tmp_path):
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")

        history, best_acc = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=2,
            patience=5
        )
        assert isinstance(history, dict)
        assert isinstance(best_acc, float)

    def test_history_keys(self, tiny_loaders, training_components, tmp_path):
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")

        history, _ = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=2,
            patience=5
        )
        expected_keys = {'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'}
        assert set(history.keys()) == expected_keys

    def test_history_length_matches_epochs(self, tiny_loaders, training_components, tmp_path):
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")
        epochs = 3

        history, _ = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=epochs,
            patience=10  # High patience so no early stopping
        )
        assert len(history['train_loss']) == epochs
        assert len(history['val_loss']) == epochs

    def test_saves_best_model(self, tiny_loaders, training_components, tmp_path):
        """Best model checkpoint should be saved to disk."""
        import os
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")

        train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=2,
            patience=5
        )
        assert os.path.exists(save_path)

    def test_saved_checkpoint_has_required_keys(self, tiny_loaders, training_components, tmp_path):
        """The saved checkpoint should contain model_state_dict and val_accuracy."""
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")

        train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=2,
            patience=5
        )
        checkpoint = torch.load(save_path, weights_only=False)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'val_accuracy' in checkpoint
        assert 'epoch' in checkpoint

    def test_early_stopping(self, training_components, tmp_path):
        """Training should stop early when val loss doesn't improve for `patience` epochs."""
        from src.fire_detection.train import train_model

        model, criterion, optimizer, scheduler = training_components

        # Create data that will produce consistent (non-improving) loss
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1])
        train_loader = DataLoader(TensorDataset(x, y), batch_size=2)
        val_loader = DataLoader(TensorDataset(x, y), batch_size=2)

        save_path = str(tmp_path / "best.pth")
        history, _ = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=100,  # Very many epochs
            patience=3   # Should stop much earlier
        )
        # Training should have stopped before 100 epochs
        assert len(history['train_loss']) < 100

    def test_accuracy_between_0_and_1(self, tiny_loaders, training_components, tmp_path):
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")

        history, best_acc = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=2,
            patience=5
        )
        assert 0.0 <= best_acc <= 1.0
        for acc in history['train_accuracy']:
            assert 0.0 <= acc <= 1.0
        for acc in history['val_accuracy']:
            assert 0.0 <= acc <= 1.0

    def test_loss_is_non_negative(self, tiny_loaders, training_components, tmp_path):
        from src.fire_detection.train import train_model
        model, criterion, optimizer, scheduler = training_components
        train_loader, val_loader = tiny_loaders
        save_path = str(tmp_path / "best.pth")

        history, _ = train_model(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            device=torch.device("cpu"),
            save_path=save_path,
            epochs=2,
            patience=5
        )
        for loss in history['train_loss']:
            assert loss >= 0
        for loss in history['val_loss']:
            assert loss >= 0
