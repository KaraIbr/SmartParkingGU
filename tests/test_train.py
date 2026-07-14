"""Tests para el modulo train.py."""

import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import SmartParkingCNN
from src.train import EarlyStopping, train_model


class TestEarlyStopping:
    """Tests para la clase EarlyStopping."""

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, mode='min')
        model = SmartParkingCNN()
        # Simular scores que no mejoran
        for _ in range(5):
            es(1.0, model)
        assert es.early_stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, mode='min')
        model = SmartParkingCNN()
        es(1.0, model)
        es(1.0, model)
        es(0.5, model)  # Improvement
        assert es.counter == 0
        assert not es.early_stop

    def test_max_mode(self):
        es = EarlyStopping(patience=2, mode='max')
        model = SmartParkingCNN()
        es(0.5, model)
        es(0.3, model)  # Worse in max mode
        es(0.1, model)  # Worse
        assert es.early_stop


class TestTrainModel:
    """Tests para train_model."""

    def _make_dummy_loaders(self, n_samples: int = 32):
        """Crea DataLoaders dummy para testing."""
        images = torch.randn(n_samples, 3, 150, 150)
        labels = torch.randint(0, 2, (n_samples,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        return loader

    def test_training_runs(self):
        """Verifica que el entrenamiento completa al menos 1 epoca."""
        model = SmartParkingCNN(num_classes=2)
        train_loader = self._make_dummy_loaders(32)
        val_loader = self._make_dummy_loaders(16)

        with tempfile.TemporaryDirectory() as tmp:
            history = train_model(
                model, train_loader, val_loader,
                num_epochs=2, learning_rate=1e-3,
                device='cpu', save_dir=tmp, patience=5,
            )
            assert 'train_loss' in history
            assert 'val_loss' in history
            assert len(history['train_loss']) >= 1
            # Verify checkpoint was saved
            assert os.path.exists(os.path.join(tmp, 'final_model.pt'))

    def test_early_stopping_works(self):
        """Verifica que early stopping detiene el entrenamiento."""
        model = SmartParkingCNN(num_classes=2)
        train_loader = self._make_dummy_loaders(32)
        val_loader = self._make_dummy_loaders(16)

        with tempfile.TemporaryDirectory() as tmp:
            history = train_model(
                model, train_loader, val_loader,
                num_epochs=100, learning_rate=1e-3,
                device='cpu', save_dir=tmp, patience=2,
            )
            # Should stop before 100 epochs
            assert len(history['train_loss']) < 100
