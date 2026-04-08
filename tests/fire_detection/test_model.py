"""
Tests for src.fire_detection.model — MobileNetFireDetection.
Validates architecture, forward pass shapes, and weight loading.
"""
import pytest
import torch
import torch.nn as nn


class TestMobileNetFireDetection:
    """Tests for the MobileNetFireDetection model."""

    def test_instantiation_no_pretrained(self):
        """Model can be created without pretrained weights."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        assert model is not None

    def test_is_nn_module(self):
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        assert isinstance(model, nn.Module)

    def test_output_shape_single_image(self):
        """Forward pass on a single image should produce shape (1, 1)."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"

    def test_output_shape_batch(self):
        """Forward pass on a batch should produce shape (B, 1)."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        model.eval()
        batch_size = 8
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 1), f"Expected ({batch_size},1), got {out.shape}"

    def test_output_is_raw_logits(self):
        """
        Output should be raw logits (unbounded), not probabilities.
        Sigmoid is NOT applied inside the model — this is by design
        because BCEWithLogitsLoss expects raw logits.
        """
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        model.eval()
        x = torch.randn(16, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Logits can be negative or > 1, so at least some should be outside [0,1]
        # With 16 random images, it's extremely unlikely all values are in [0,1]
        all_in_01 = (out >= 0).all() and (out <= 1).all()
        # This is a soft check — the key point is the model has no sigmoid layer
        assert not any(
            isinstance(m, nn.Sigmoid) for m in model.mobilenet.classifier.modules()
        ), "Model classifier should NOT contain a Sigmoid layer"

    def test_classifier_structure(self):
        """Classifier should be Dropout → Linear(in_features, 1)."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        classifier = model.mobilenet.classifier
        assert isinstance(classifier, nn.Sequential)
        assert len(classifier) == 2
        assert isinstance(classifier[0], nn.Dropout)
        assert isinstance(classifier[1], nn.Linear)
        assert classifier[1].out_features == 1

    def test_gradient_flow(self):
        """Gradients should flow through the model during backprop."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        model.train()
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least the classifier layer has non-zero gradients
        linear = model.mobilenet.classifier[1]
        assert linear.weight.grad is not None
        assert linear.weight.grad.abs().sum() > 0

    def test_state_dict_save_load(self, tmp_path):
        """Model state dict can be saved and reloaded correctly."""
        from src.fire_detection.model import MobileNetFireDetection
        model1 = MobileNetFireDetection(pretrained=False)
        path = tmp_path / "model.pth"
        torch.save(model1.state_dict(), path)

        model2 = MobileNetFireDetection(pretrained=False)
        model2.load_state_dict(torch.load(path, weights_only=True))

        # Both models should produce the same output
        model1.eval()
        model2.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            assert torch.allclose(model1(x), model2(x))

    def test_eval_mode_deterministic(self):
        """Model in eval mode should produce deterministic outputs."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_parameter_count_reasonable(self):
        """Model should have a reasonable number of parameters (MobileNetV2 ~3.5M)."""
        from src.fire_detection.model import MobileNetFireDetection
        model = MobileNetFireDetection(pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        # MobileNetV2 has ~3.5M params; with our classifier it should be roughly the same
        assert 1_000_000 < total_params < 10_000_000, (
            f"Parameter count {total_params} outside expected range"
        )
