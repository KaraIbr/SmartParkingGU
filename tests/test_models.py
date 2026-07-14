"""Tests para el modulo models.py."""

import pytest
import torch

from scripts.models import SmartParkingCNN, get_model_info


class TestSmartParkingCNN:
    """Tests para la arquitectura SmartParkingCNN."""

    def test_output_shape_batch1(self):
        """Verifica que la salida tiene forma correcta con batch=1."""
        model = SmartParkingCNN(num_classes=2, dropout_rate=0.5)
        dummy = torch.randn(1, 3, 150, 150)
        output = model(dummy)
        assert output.shape == (1, 2)

    def test_output_shape_batch32(self):
        """Verifica que la salida tiene forma correcta con batch=32."""
        model = SmartParkingCNN(num_classes=2)
        dummy = torch.randn(32, 3, 150, 150)
        output = model(dummy)
        assert output.shape == (32, 2)

    def test_num_classes(self):
        """Verifica que num_classes afecta la salida."""
        model = SmartParkingCNN(num_classes=3)
        dummy = torch.randn(1, 3, 150, 150)
        output = model(dummy)
        assert output.shape == (1, 3)

    def test_dropout_rate(self):
        """Verifica que dropout se aplica correctamente."""
        model_train = SmartParkingCNN(dropout_rate=0.5)
        model_train.train()
        dummy = torch.randn(8, 3, 150, 150)
        out1 = model_train(dummy)
        out2 = model_train(dummy)
        # Con dropout activo, salidas deben ser diferentes
        assert not torch.allclose(out1, out2)

        model_eval = SmartParkingCNN(dropout_rate=0.5)
        model_eval.load_state_dict(model_train.state_dict())
        model_eval.eval()
        out3 = model_eval(dummy)
        out4 = model_eval(dummy)
        # Sin dropout, salidas deben ser iguales
        assert torch.allclose(out3, out4)

    def test_count_parameters(self):
        """Verifica que count_parameters retorna un entero positivo."""
        model = SmartParkingCNN()
        params = model.count_parameters()
        assert isinstance(params, int)
        assert params > 0
        # Debe tener ~4.7M parametros
        assert params > 1_000_000

    def test_gradients_flow(self):
        """Verifica que los gradientes fluyen correctamente."""
        model = SmartParkingCNN()
        dummy = torch.randn(2, 3, 150, 150)
        labels = torch.tensor([0, 1])
        output = model(dummy)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestGetModelInfo:
    """Tests para get_model_info."""

    def test_returns_dict(self):
        model = SmartParkingCNN()
        info = get_model_info(model)
        assert isinstance(info, dict)

    def test_expected_keys(self):
        model = SmartParkingCNN()
        info = get_model_info(model)
        assert 'arquitectura' in info
        assert 'parametros_totales' in info
        assert 'parametros_miles' in info

    def test_architecture_name(self):
        model = SmartParkingCNN()
        info = get_model_info(model)
        assert info['arquitectura'] == 'SmartParkingCNN'
