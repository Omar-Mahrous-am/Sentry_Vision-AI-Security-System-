"""
Tests for src.fire_detection.preprocessing — get_mean_std and augmentation_main_transform.
Validates computed statistics and transform pipelines.
"""
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class TestGetMeanStd:
    """Tests for dataset mean/std computation."""

    @pytest.fixture
    def simple_image_dataset(self):
        """Create a tiny in-memory dataset of PIL images and labels."""
        images = []
        for _ in range(10):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            images.append((Image.fromarray(arr), 0))
        return images

    def test_returns_two_tensors(self, simple_image_dataset):
        from src.fire_detection.preprocessing import get_mean_std
        mean, std = get_mean_std(simple_image_dataset)
        assert isinstance(mean, torch.Tensor)
        assert isinstance(std, torch.Tensor)

    def test_shape_is_3(self, simple_image_dataset):
        """Mean and std should each have 3 values (one per channel)."""
        from src.fire_detection.preprocessing import get_mean_std
        mean, std = get_mean_std(simple_image_dataset)
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_mean_in_valid_range(self, simple_image_dataset):
        """After ToTensor (0-1 range), mean should be in [0, 1]."""
        from src.fire_detection.preprocessing import get_mean_std
        mean, std = get_mean_std(simple_image_dataset)
        for m in mean:
            assert 0.0 <= m.item() <= 1.0

    def test_std_positive(self, simple_image_dataset):
        """Standard deviation should be positive."""
        from src.fire_detection.preprocessing import get_mean_std
        mean, std = get_mean_std(simple_image_dataset)
        for s in std:
            assert s.item() > 0

    def test_all_black_images_mean_zero(self):
        """For all-black images, mean should be approximately 0."""
        from src.fire_detection.preprocessing import get_mean_std
        black_imgs = [(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)), 0) for _ in range(5)]
        mean, std = get_mean_std(black_imgs)
        assert torch.allclose(mean, torch.zeros(3), atol=1e-5)

    def test_all_white_images_mean_one(self):
        """For all-white images (255), mean should be approximately 1.0."""
        from src.fire_detection.preprocessing import get_mean_std
        white_imgs = [(Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8)), 0) for _ in range(5)]
        mean, std = get_mean_std(white_imgs)
        assert torch.allclose(mean, torch.ones(3), atol=1e-5)

    def test_raises_on_empty_dataset(self):
        """Should raise ValueError if no valid images are found."""
        from src.fire_detection.preprocessing import get_mean_std
        with pytest.raises(ValueError, match="No valid images"):
            get_mean_std([])

    def test_skips_non_image_entries(self):
        """String paths and non-Image types should be skipped gracefully."""
        from src.fire_detection.preprocessing import get_mean_std
        mixed = [
            ("/path/to/video.mp4", 0),  # String path — should be skipped
            (Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)), 1),
        ]
        mean, std = get_mean_std(mixed)
        # Should succeed with the single valid image
        assert mean.shape == (3,)


class TestAugmentationMainTransform:
    """Tests for the augmentation_main_transform function."""

    @pytest.fixture
    def mean_std(self):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    def test_returns_three_transforms(self, mean_std):
        from src.fire_detection.preprocessing import augmentation_main_transform
        main_t, fire_t, nofire_t = augmentation_main_transform(*mean_std)
        assert main_t is not None
        assert fire_t is not None
        assert nofire_t is not None

    def test_main_transform_produces_tensor(self, mean_std, dummy_pil_image):
        from src.fire_detection.preprocessing import augmentation_main_transform
        main_t, _, _ = augmentation_main_transform(*mean_std)
        result = main_t(dummy_pil_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_fire_transform_produces_tensor(self, mean_std, dummy_pil_image):
        from src.fire_detection.preprocessing import augmentation_main_transform
        _, fire_t, _ = augmentation_main_transform(*mean_std)
        result = fire_t(dummy_pil_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_nofire_transform_produces_tensor(self, mean_std, dummy_pil_image):
        from src.fire_detection.preprocessing import augmentation_main_transform
        _, _, nofire_t = augmentation_main_transform(*mean_std)
        result = nofire_t(dummy_pil_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)

    def test_main_transform_is_deterministic(self, mean_std, dummy_pil_image):
        """Val/test transform should produce identical results each time."""
        from src.fire_detection.preprocessing import augmentation_main_transform
        main_t, _, _ = augmentation_main_transform(*mean_std)
        r1 = main_t(dummy_pil_image)
        r2 = main_t(dummy_pil_image)
        assert torch.allclose(r1, r2)

    def test_train_transforms_include_augmentation(self, mean_std):
        """
        The fire and no-fire training transforms should contain
        RandomHorizontalFlip, confirming augmentation is present.
        """
        from src.fire_detection.preprocessing import augmentation_main_transform
        _, fire_t, nofire_t = augmentation_main_transform(*mean_std)
        fire_transform_types = [type(t).__name__ for t in fire_t.transforms]
        nofire_transform_types = [type(t).__name__ for t in nofire_t.transforms]
        assert "RandomHorizontalFlip" in fire_transform_types
        assert "RandomHorizontalFlip" in nofire_transform_types
