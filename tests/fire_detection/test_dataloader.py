"""
Tests for src.fire_detection.dataloader — get_dataloaders function.
Validates correct splitting, loader sizes, and batch shapes.
"""
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class TestGetDataloaders:
    """Tests for the get_dataloaders factory function."""

    @pytest.fixture
    def dataset_and_transforms(self, fire_image_dir):
        """Create a Fire_Dataset and transforms for testing."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=None)
        main_t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return ds, main_t, main_t, main_t

    def test_returns_three_loaders(self, dataset_and_transforms):
        from src.fire_detection.dataloader import get_dataloaders
        ds, main_t, fire_t, nofire_t = dataset_and_transforms
        result = get_dataloaders(
            ds, batch_size=2,
            train_fac=0.5, val_fac=0.25, test_fac=0.25,
            main_transform=main_t,
            fire_train_transform=fire_t,
            no_fire_train_transform=nofire_t
        )
        assert len(result) == 3
        train_loader, val_loader, test_loader = result
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_split_sizes_correct(self, dataset_and_transforms):
        """The total samples across loaders should equal the original dataset size."""
        from src.fire_detection.dataloader import get_dataloaders
        ds, main_t, fire_t, nofire_t = dataset_and_transforms
        total = len(ds)
        train_loader, val_loader, test_loader = get_dataloaders(
            ds, batch_size=1,
            train_fac=0.5, val_fac=0.25, test_fac=0.25,
            main_transform=main_t,
            fire_train_transform=fire_t,
            no_fire_train_transform=nofire_t
        )
        # Count samples yielded by each loader
        train_count = sum(len(batch[0]) for batch in train_loader)
        val_count = sum(len(batch[0]) for batch in val_loader)
        test_count = sum(len(batch[0]) for batch in test_loader)
        assert train_count + val_count + test_count == total

    def test_batch_shape(self, dataset_and_transforms):
        """Each batch should have shape (B, 3, 224, 224)."""
        from src.fire_detection.dataloader import get_dataloaders
        ds, main_t, fire_t, nofire_t = dataset_and_transforms
        train_loader, _, _ = get_dataloaders(
            ds, batch_size=2,
            train_fac=0.5, val_fac=0.25, test_fac=0.25,
            main_transform=main_t,
            fire_train_transform=fire_t,
            no_fire_train_transform=nofire_t
        )
        for images, labels in train_loader:
            assert images.dim() == 4
            assert images.shape[1] == 3
            assert images.shape[2] == 224
            assert images.shape[3] == 224
            break  # Only check first batch

    def test_labels_are_valid(self, dataset_and_transforms):
        """All labels should be 0 or 1."""
        from src.fire_detection.dataloader import get_dataloaders
        ds, main_t, fire_t, nofire_t = dataset_and_transforms
        train_loader, val_loader, test_loader = get_dataloaders(
            ds, batch_size=4,
            train_fac=0.5, val_fac=0.25, test_fac=0.25,
            main_transform=main_t,
            fire_train_transform=fire_t,
            no_fire_train_transform=nofire_t
        )
        for loader in [train_loader, val_loader, test_loader]:
            for _, labels in loader:
                for l in labels:
                    assert l.item() in (0, 1), f"Unexpected label {l.item()}"
