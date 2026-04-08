"""
Shared pytest fixtures for the Sentry Vision test suite.
Provides reusable dummy data, models, and temporary directories
that are used across multiple fire_detection test modules.
"""
import pytest
import torch
import numpy as np
import os
import tempfile
from PIL import Image
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory (pytest built-in tmp_path wrapper)."""
    return tmp_path


# ---------------------------------------------------------------------------
# Dummy images
# ---------------------------------------------------------------------------
@pytest.fixture
def dummy_pil_image():
    """Return a small 224x224 RGB PIL Image."""
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))


@pytest.fixture
def dummy_tensor_image():
    """Return a 3x224x224 normalised float tensor (single image)."""
    return torch.randn(3, 224, 224)


@pytest.fixture
def dummy_batch():
    """Return a batch of 4 images with binary labels."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 1, 0])
    return images, labels


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def fire_image_dir(tmp_path):
    """
    Build a minimal on-disk directory tree that Fire_Dataset can parse.
    Structure:
        tmp_path/
        └── img_data/
            ├── fire_images/
            │   ├── fire_001.jpg
            │   └── fire_002.jpg
            ├── default/
            │   ├── nofire_001.jpg
            │   └── nofire_002.jpg
    """
    fire_dir = tmp_path / "img_data" / "fire_images"
    fire_dir.mkdir(parents=True)
    nofire_dir = tmp_path / "img_data" / "default"
    nofire_dir.mkdir(parents=True)

    for i in range(2):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(fire_dir / f"fire_{i:03d}.jpg")
        img.save(nofire_dir / f"nofire_{i:03d}.jpg")

    return tmp_path


@pytest.fixture
def video_dir(tmp_path):
    """
    Build a minimal on-disk directory tree with video stubs.
    Uses cv2 to write a tiny MP4 so _get_frame_count / _load_frame work.
    Falls back to an empty dir if cv2 is unavailable.
    """
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available — skipping video tests")

    train_dir = tmp_path / "video_data" / "train_videos"
    train_dir.mkdir(parents=True)
    test_dir = tmp_path / "video_data" / "test_videos"
    test_dir.mkdir(parents=True)

    # Write a tiny 5-frame video
    for name in ["fire_clip.mp4", "nofire_clip.mp4"]:
        path = str(train_dir / name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10, (64, 64))
        for _ in range(5):
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    # Write an unlabeled test video
    test_path = str(test_dir / "unknown_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(test_path, fourcc, 10, (64, 64))
    for _ in range(5):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return tmp_path


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def fire_model():
    """Return a MobileNetFireDetection model (non-pretrained for speed)."""
    from src.fire_detection.model import MobileNetFireDetection
    return MobileNetFireDetection(pretrained=False)


@pytest.fixture
def saved_model_path(tmp_path, fire_model):
    """Save a model checkpoint to disk and return the path."""
    save_path = tmp_path / "test_model.pth"
    torch.save({
        'model_state_dict': fire_model.state_dict(),
        'optimizer_state_dict': {},
        'val_accuracy': 0.95,
        'val_loss': 0.1,
        'epoch': 1
    }, save_path)
    return str(save_path)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_dataloader(dummy_batch):
    """Return a single-batch DataLoader for quick evaluation tests."""
    images, labels = dummy_batch
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=4)


@pytest.fixture
def device():
    """Return the computation device (CPU in tests)."""
    return torch.device("cpu")
