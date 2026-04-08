"""
Tests for src.fire_detection.config — Config class.
Validates defaults, types, and invariants of the configuration object.
"""
import pytest
import torch


class TestConfig:
    """Tests for the Config singleton."""

    def test_import(self):
        """Config can be imported without errors."""
        from src.fire_detection.config import Config, config
        assert config is not None
        assert isinstance(config, Config)

    def test_data_paths_are_strings(self):
        from src.fire_detection.config import config
        assert isinstance(config.DATA_DIR, str)
        assert isinstance(config.MODELS_DIR, str)
        assert isinstance(config.MODEL_PATH, str)
        assert isinstance(config.BEST_MODEL_PATH, str)

    def test_model_path_inside_models_dir(self):
        """MODEL_PATH and BEST_MODEL_PATH should start with MODELS_DIR."""
        from src.fire_detection.config import config
        assert config.MODEL_PATH.startswith(config.MODELS_DIR)
        assert config.BEST_MODEL_PATH.startswith(config.MODELS_DIR)

    def test_split_ratios_sum_to_one(self):
        from src.fire_detection.config import config
        total = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
        assert abs(total - 1.0) < 1e-6, f"Splits must sum to 1.0, got {total}"

    def test_split_ratios_positive(self):
        from src.fire_detection.config import config
        assert config.TRAIN_SPLIT > 0
        assert config.VAL_SPLIT > 0
        assert config.TEST_SPLIT > 0

    def test_hyperparameter_types(self):
        from src.fire_detection.config import config
        assert isinstance(config.BATCH_SIZE, int)
        assert isinstance(config.EPOCHS, int)
        assert isinstance(config.PATIENCE, int)
        assert isinstance(config.LEARNING_RATE, float)
        assert isinstance(config.THRESHOLD, float)
        assert isinstance(config.FRAMES_PER_VIDEO, int)

    def test_hyperparameter_ranges(self):
        from src.fire_detection.config import config
        assert config.BATCH_SIZE > 0
        assert config.EPOCHS > 0
        assert config.PATIENCE > 0
        assert 0 < config.LEARNING_RATE < 1
        assert 0 < config.THRESHOLD < 1
        assert config.FRAMES_PER_VIDEO > 0

    def test_img_size_is_tuple(self):
        from src.fire_detection.config import config
        assert isinstance(config.IMG_SIZE, tuple)
        assert len(config.IMG_SIZE) == 2
        assert all(s > 0 for s in config.IMG_SIZE)

    def test_num_classes(self):
        from src.fire_detection.config import config
        assert config.NUM_CLASSES == 1  # Binary classification with BCEWithLogitsLoss

    def test_device_is_torch_device(self):
        from src.fire_detection.config import config
        assert isinstance(config.DEVICE, torch.device)

    def test_normalization_stats_length(self):
        from src.fire_detection.config import config
        assert len(config.NORM_MEAN) == 3
        assert len(config.NORM_STD) == 3

    def test_normalization_stats_values_in_range(self):
        """ImageNet stats should be in [0, 1]."""
        from src.fire_detection.config import config
        for m in config.NORM_MEAN:
            assert 0.0 <= m <= 1.0
        for s in config.NORM_STD:
            assert 0.0 < s <= 1.0
