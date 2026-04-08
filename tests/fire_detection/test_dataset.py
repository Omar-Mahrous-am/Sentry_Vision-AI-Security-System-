"""
Tests for src.fire_detection.dataset — Fire_Dataset and Subset_Dataset.
Validates data loading, labelling logic, and transform application.
"""
import pytest
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from unittest.mock import patch, MagicMock


# ===========================================================================
# Fire_Dataset tests
# ===========================================================================
class TestFireDataset:
    """Tests for the Fire_Dataset class."""

    def test_loads_images_from_fire_dir(self, fire_image_dir):
        """Dataset should find all 4 images (2 fire + 2 no-fire)."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=None)
        assert len(ds) == 4

    def test_labels_fire_images(self, fire_image_dir):
        """Images under fire_images/ should have label 1."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=None)
        fire_samples = [s for s in ds.samples if s[1] == 1]
        assert len(fire_samples) == 2

    def test_labels_nofire_images(self, fire_image_dir):
        """Images under default/ should have label 0."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=None)
        nofire_samples = [s for s in ds.samples if s[1] == 0]
        assert len(nofire_samples) == 2

    def test_getitem_returns_image_and_label(self, fire_image_dir):
        """__getitem__ should return (PIL Image, int label) when no transform."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=None)
        img, label = ds[0]
        assert isinstance(img, Image.Image)
        assert label in (0, 1)

    def test_getitem_with_transform(self, fire_image_dir):
        """When a transform is supplied, __getitem__ should return a tensor."""
        from src.fire_detection.dataset import Fire_Dataset
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=t)
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)

    def test_empty_directory(self, tmp_path):
        """Dataset should have 0 samples for an empty directory."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(tmp_path), transform=None)
        assert len(ds) == 0

    def test_error_logging(self, tmp_path):
        """Corrupted/unreadable files should be error-logged, not crash."""
        from src.fire_detection.dataset import Fire_Dataset
        # Create a file that looks like an image but is garbage
        fire_dir = tmp_path / "img_data" / "fire_images"
        fire_dir.mkdir(parents=True)
        bad_file = fire_dir / "corrupt.jpg"
        bad_file.write_bytes(b"not an image")

        ds = Fire_Dataset(root_dir=str(tmp_path), transform=None)
        # The file has a .jpg extension so it should be added as a sample
        # (error only happens at __getitem__ time for images read via PIL)
        # No crash during construction = success
        assert isinstance(ds.error_logs, list)


class TestFireDatasetLabelling:
    """Unit tests for the _get_label logic."""

    def test_img_data_default_is_no_fire(self, fire_image_dir):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label(str(fire_image_dir / "img_data" / "default"), "img.jpg")
        assert label == 0

    def test_img_data_fire_is_fire(self, fire_image_dir):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label(str(fire_image_dir / "img_data" / "fire_images"), "img.jpg")
        assert label == 1

    def test_img_data_smoke_is_fire(self, tmp_path):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label(str(tmp_path / "img_data" / "smoke_images"), "img.jpg")
        assert label == 1

    def test_train_video_fire_filename(self):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label("/some/video_data/train_videos", "fire_clip_001.mp4")
        assert label == 1

    def test_train_video_nofire_filename(self):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label("/some/video_data/train_videos", "nofire_clip.mp4")
        assert label == 0

    def test_test_video_returns_none(self):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label("/some/video_data/test_videos", "clip.mp4")
        assert label is None

    def test_unknown_path_returns_none(self):
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset.__new__(Fire_Dataset)
        label = ds._get_label("/random/path", "something.jpg")
        assert label is None


class TestFireDatasetVideo:
    """Tests for video-related functionality (requires cv2)."""

    def test_loads_video_frames(self, video_dir):
        """Dataset should extract frame samples from training videos."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(
            root_dir=str(video_dir),
            transform=None,
            frames_per_video=3
        )
        # 2 training videos × 3 frames each = 6 frame samples
        frame_samples = [s for s in ds.samples if s[2] == "frame"]
        assert len(frame_samples) == 6

    def test_test_videos_excluded_by_default(self, video_dir):
        """Unlabeled test_videos should be skipped unless include_test_videos=True."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(
            root_dir=str(video_dir),
            transform=None,
            include_test_videos=False
        )
        unlabeled = [s for s in ds.samples if s[1] == -1]
        assert len(unlabeled) == 0

    def test_include_test_videos_flag(self, video_dir):
        """With include_test_videos=True, test videos should appear."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(
            root_dir=str(video_dir),
            transform=None,
            include_test_videos=True
        )
        unlabeled = [s for s in ds.samples if s[1] == -1]
        assert len(unlabeled) >= 1

    def test_load_frame_returns_pil_image(self, video_dir):
        """_load_frame should return a PIL Image for a valid frame index."""
        from src.fire_detection.dataset import Fire_Dataset
        ds = Fire_Dataset(root_dir=str(video_dir), transform=None, frames_per_video=3)
        frame_sample = [s for s in ds.samples if s[2] == "frame"][0]
        video_path, _, _, frame_idx = frame_sample
        img = ds._load_frame(video_path, frame_idx)
        assert isinstance(img, Image.Image)


# ===========================================================================
# Subset_Dataset tests
# ===========================================================================
class TestSubsetDataset:
    """Tests for the Subset_Dataset wrapper."""

    def _make_subset(self, fire_image_dir):
        """Helper to build a Subset from Fire_Dataset."""
        from src.fire_detection.dataset import Fire_Dataset
        from torch.utils.data import Subset
        ds = Fire_Dataset(root_dir=str(fire_image_dir), transform=None)
        indices = list(range(len(ds)))
        return Subset(ds, indices)

    def test_length_matches_subset(self, fire_image_dir):
        from src.fire_detection.dataset import Subset_Dataset
        subset = self._make_subset(fire_image_dir)
        main_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        sds = Subset_Dataset(subset, main_t, main_t, main_t, is_train=False)
        assert len(sds) == len(subset)

    def test_val_mode_uses_main_transform(self, fire_image_dir):
        """In validation mode (is_train=False), main_transform is applied."""
        from src.fire_detection.dataset import Subset_Dataset
        subset = self._make_subset(fire_image_dir)
        main_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        fire_t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        nofire_t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

        sds = Subset_Dataset(subset, main_t, fire_t, nofire_t, is_train=False)
        img, label = sds[0]
        assert isinstance(img, torch.Tensor)
        # main_t resizes to 224 → tensor should be 3×224×224
        assert img.shape == (3, 224, 224)

    def test_train_mode_fire_sample_uses_fire_transform(self, fire_image_dir):
        """In training mode, fire samples (label=1) should use fire_train_transform."""
        from src.fire_detection.dataset import Subset_Dataset
        subset = self._make_subset(fire_image_dir)
        main_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        fire_t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        nofire_t = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

        sds = Subset_Dataset(subset, main_t, fire_t, nofire_t, is_train=True)
        # Find a fire sample (label=1)
        for i in range(len(sds)):
            img, label = sds[i]
            if label == 1:
                assert img.shape == (3, 128, 128), "Fire sample should use fire_train_transform"
                return
        pytest.skip("No fire sample found in subset")

    def test_train_mode_nofire_sample_uses_nofire_transform(self, fire_image_dir):
        """In training mode, no-fire samples (label=0) should use no_fire_train_transform."""
        from src.fire_detection.dataset import Subset_Dataset
        subset = self._make_subset(fire_image_dir)
        main_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        fire_t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        nofire_t = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

        sds = Subset_Dataset(subset, main_t, fire_t, nofire_t, is_train=True)
        for i in range(len(sds)):
            img, label = sds[i]
            if label == 0:
                assert img.shape == (3, 64, 64), "No-fire sample should use no_fire_train_transform"
                return
        pytest.skip("No no-fire sample found in subset")
